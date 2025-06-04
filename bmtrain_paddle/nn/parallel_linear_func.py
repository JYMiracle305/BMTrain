import paddle
import paddle.nn.functional as F
import bmtrain_paddle as bmt
from bmtrain_paddle.global_var import config
from ..distributed import all_gather, all_reduce
from .. import nccl
from enum import Enum


class ReduceType(Enum):
    ALL_REDUCE = 1
    REDUCE_SCATTER = 2


def preprocess_input(input, gather_input, split_input):
    if gather_input:
        input = all_gather(input, config["tp_comm"])
        input = input.flatten(0, 1)

    if split_input:
        all_input_list = input.chunk(config["tp_size"], dim=-1)
        input = all_input_list[config["topology"].tp_id]
    return input


def async_all_gather_linear_func(input, weight, bias, async_chunks=2):
    # print("---------------async_all_gather_linear_func before", input.shape, weight.shape, bias.shape)
    dim = input.dim()
    shape = list(input.shape)
    if dim > 2:
        input = input.reshape([-1, input.shape[-1]])
    tp_size = config["tp_size"]
    current_stream = paddle.device.cuda.current_stream()
    comm_stream = config["tp_comm_stream"]

    rounds = async_chunks
    inputs = input.chunk(rounds, axis=0)
    comm_stream.wait_stream(current_stream)
    outputs = [None] * tp_size * rounds
    # print("--------------- 11111111111 async_all_gather_linear_func", input)
    input = all_gather(inputs[0], config["tp_comm"])
    # print("--------------- 22222222222 async_all_gather_linear_func", input.shape)

    input = input.flatten(0, 1)
    # print("--------------- 33333333333 async_all_gather_linear_func", input, weight)
    # 检查输入和权重的数值范围
    # print("input max/min:", input.max().item(), input.min().item())
    # print("weight max/min:", weight.max().item(), weight.min().item())

    out = F.linear(input, weight, bias)
    # print("--------------- 44444444444 async_all_gather_linear_func", out)
    outs = out.chunk(tp_size, axis=0)
    # print("---------------async_all_gather_linear_func after after", input, outs)
    for i in range(tp_size):
        outputs[i * rounds] = outs[i]

    # async all_gather and overalap with linear
    for i in range(rounds - 1):
        with paddle.device.cuda.stream_guard(comm_stream):
            # inputs[i + 1].record_stream(comm_stream)
            input = all_gather(inputs[i + 1], config["tp_comm"])
            input = input.flatten(0, 1)

        current_stream.wait_stream(comm_stream)
        out = F.linear(input, weight, bias)
        outs = out.chunk(tp_size, axis=0)
        for j in range(tp_size):
            outputs[(i + 1) + j * rounds] = outs[j]
        # print("---------------async_all_gather_linear_func after after !!!!!!!", input, outputs)
    out = paddle.concat(outputs, axis=0)
    if dim > 2:
        out_shape = list(out.shape)
        shape[-1] = out_shape[-1]
        shape[0] = shape[0] * tp_size
        out = out.reshape(shape)

    # print("---------------async_all_gather_linear_func res", out)

    return out


def async_reduce_scatter_linear_func(input, weight, bias, async_chunks=2):
    tp_size = config["tp_size"]
    comm_stream = config["tp_comm_stream"]
    rounds = async_chunks
    input_shape = list(input.shape)
    dim = input.dim()
    if dim > 2:
        input = input.reshape([-1, input.shape[-1]])
    inputs = input.chunk(rounds * tp_size, axis=0)
    current_stream = paddle.device.cuda.current_stream()

    outputs = [None] * rounds
    for i in range(rounds):
        input = [None] * tp_size
        for j in range(tp_size):
            input[j] = inputs[j * rounds + i]
        input = paddle.concat(input, axis=0)
        out = F.linear(input, weight, bias)
        with paddle.device.cuda.stream_guard(comm_stream):
            comm_stream.wait_stream(current_stream)
            # out.record_stream(comm_stream)
            shape = list(out.shape)
            shape[0] = shape[0] // config["tp_size"]
            # print("async_reduce_scatter_linear_func shape[0]", shape[0])
            outputs[i] = paddle.empty(shape, dtype=out.dtype)
            # print("async_reduce_scatter_linear_func out outputs[0]", out.shape, outputs[i].shape,
            #       out.numel().item(), outputs[i].numel().item())
            if out.place.is_gpu_place():
                outputs[i] = outputs[i].cuda()
            nccl.reduceScatter(
                out, outputs[i], "sum", config["tp_comm"]
            )

    current_stream.wait_stream(comm_stream)
    out = paddle.concat(outputs, axis=0)
    if dim > 2:
        out_shape = list(out.shape)
        input_shape[-1] = out_shape[-1]
        input_shape[0] = input_shape[0] // tp_size
        out = out.view(input_shape)

    return out


def async_all_gather_linear_backward_func(
    grad_out, input, weight, bias, async_chunks=2
):
    tp_size = config["tp_size"]
    current_stream = paddle.device.cuda.current_stream()
    comm_stream = config["tp_comm_stream"]
    input_require_grad = not input.stop_gradient
    dim = input.dim()
    input_shape = input.shape
    if dim > 2:
        input = input.view([-1, input_shape[-1]])
        grad_out = grad_out.view([-1, grad_out.shape[-1]])

    rounds = async_chunks
    grad_inputs = [None] * tp_size * rounds
    grad_weights = [None] * tp_size * rounds
    grad_outs = [None] * tp_size * rounds
    local_grad_outs = grad_out.chunk(rounds, dim=0)

    inputs = [None] * rounds
    comm_stream.wait_stream(current_stream)
    if not weight.stop_gradient:
        with paddle.device.cuda.stream_guard(comm_stream):
            input.record_stream(comm_stream)
            input_list = [None] * tp_size * rounds
            tp_inputs = input.chunk(tp_size, dim=0)
            for i in range(tp_size):
                chunk_inputs = tp_inputs[i].chunk(rounds, dim=0)
                for j in range(rounds):
                    input_list[j * tp_size + i] = chunk_inputs[j]
            start = 0
            end = tp_size
            for i in range(rounds):
                inputs[i] = paddle.cat(input_list[start:end], dim=0)
                start = end
                end += tp_size

    grad_input = grad_weight = grad_bias = None

    grad_out = all_gather(local_grad_outs[0], config["tp_comm"])
    for j in range(tp_size):
        grad_outs[j * rounds] = grad_out[j]
    grad_out = grad_out.flatten(0, 1)  # (tp_size * (m/rounds), n)
    if input_require_grad:
        grad_input = grad_out.matmul(
            weight
        )  # (tp_size * (m/rounds), n) * (n, k/tp_size)
        tmp_grad_inputs = grad_input.chunk(tp_size, dim=0)
        for j in range(tp_size):
            grad_inputs[j * rounds] = tmp_grad_inputs[j]

    if not weight.stop_gradient:
        grad_weight = (
            grad_out.reshape([-1, grad_out.shape[-1]])
            .t()
            .matmul(inputs[0].reshape([-1, inputs[0].shape[-1]]))
        )

    # async all_gather and overalap with matmul
    for i in range(rounds - 1):
        with paddle.device.cuda.stream_guard(comm_stream):
            local_grad_outs[i + 1].record_stream(comm_stream)
            grad_out = all_gather(local_grad_outs[i + 1], config["tp_comm"])
            for j in range(tp_size):
                grad_outs[j * rounds + i + 1] = grad_out[j]
            grad_out = grad_out.flatten(0, 1)  # (tp_size * (m/rounds), n)

        current_stream.wait_stream(comm_stream)
        if input_require_grad:
            grad_input = grad_out.matmul(
                weight
            )  # (tp_size * (m/rounds), n) * (n, k/tp_size)
            tmp_grad_inputs = grad_input.chunk(tp_size, dim=0)
            for j in range(tp_size):
                grad_inputs[j * rounds + i + 1] = tmp_grad_inputs[j]

        if not weight.stop_gradient:
            dim = grad_out.dim()
            grad_weight += (
                grad_out.reshape(-1, grad_out.shape[-1])
                .t()
                .matmul(inputs[i + 1].reshape(-1, inputs[i + 1].shape[-1]))
            )

    if input_require_grad:
        grad_input = paddle.cat(grad_inputs, dim=0)
        grad_input = grad_input.view(input_shape)

    if bias is not None and not bias.stop_gradient:
        grad_out = paddle.cat(grad_outs, dim=0)
        grad_bias = grad_out.reshape([-1, grad_out.shape[-1]]).sum(0)

    return grad_input, grad_weight, grad_bias


class OpParallelLinear(paddle.autograd.PyLayer):
    """OpParallelLinear is a subclass of torch.autograd.Function.
    It gathers the input tensor when needed, and all reduce or reduece scatter the output when needed.

    """

    @staticmethod
    def forward(
        ctx,
        input,
        weight,
        bias=None,
        gather_input=False,
        gather_output=False,
        split_input=False,
        reduce_output_type=None,
        async_gather_chunks=2,
    ):
        # print("OpParallelLinear OpParallelLinear OpParallelLinear OpParallelLinear forward",
        #       input, weight, bias, gather_input, gather_output, split_input,
        #       reduce_output_type, async_gather_chunks)
        # num_args = len([input, weight, bias, gather_input, gather_output, split_input, reduce_output_type, async_gather_chunks])
        # print(f"OpParallelLinear.forward 参数数量: {num_args}") 
        if reduce_output_type is not None:
            reduce_output_type = ReduceType(reduce_output_type)
        # print("OpParallelLinear forward", input, weight, bias)
        ctx.save_for_backward(input, weight, bias)
        ctx.gather_output = gather_output
        ctx.split_input = split_input
        ctx.gather_input = gather_input
        ctx.reduce_output_type = reduce_output_type
        ctx.async_gather_chunks = async_gather_chunks

        if (
            gather_input
            and config["tp_size"] > 1
            and async_gather_chunks > 1
            and split_input == False
        ):
            out = async_all_gather_linear_func(input, weight, bias, async_gather_chunks)
        elif reduce_output_type == ReduceType.REDUCE_SCATTER:
            return async_reduce_scatter_linear_func(
                input, weight, bias, async_gather_chunks
            )
        else:
            all_input = preprocess_input(input, ctx.gather_input, ctx.split_input)
            # print(f"!!!!!!!!!!!!!!!!!!!!! input, weight, ctx.split_input \
            #       {all_input.shape} {weight.shape} {bias}")
            out = F.linear(all_input, weight, bias)
            # print(f"OpParallelLinear F.linear {out.shape}")
        if gather_output:
            all_output_list = all_gather(out, config["tp_comm"])
            all_output_list = all_output_list.chunk(config["tp_size"], dim=0)
            out = paddle.concat(all_output_list, axis=all_output_list[0].dim() - 1).flatten(
                0, 1
            )

        if reduce_output_type is None:
            return out

        if reduce_output_type == ReduceType.ALL_REDUCE:
            nccl.allReduce(out, out, "sum", config["tp_comm"])
            return out
        else:
            assert False, "no support reduce type{}".format(reduce_output_type)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensor()
        gather_output = ctx.gather_output
        print("-----------------OpParallelLinear backward ctx.saved_tensor()------------", input, weight, bias)
        if ctx.reduce_output_type == ReduceType.REDUCE_SCATTER:
            # print("———————--------OpParallelLinear backward------------———————1",
            #       input.stop_gradient, weight.stop_gradient)
            if (not input.stop_gradient) or (not weight.stop_gradient):
                grad_input, grad_weight, grad_bias = (
                    async_all_gather_linear_backward_func(
                        grad_output, input, weight, bias, ctx.async_gather_chunks
                    )
                )
                # return grad_input, grad_weight, grad_bias, None, None, None, None, None
                return (grad_input, grad_weight, grad_bias)
            else:
                grad_output = all_gather(grad_output, config["tp_comm"])
                grad_output = grad_output.flatten(0, 1)

        if gather_output:
            tp_size = config["tp_size"]
            tp_id = config["topology"].tp_id
            grad_output_list = grad_output.chunk(tp_size, dim=-1)
            grad_output = grad_output_list[tp_id]

        grad_input = grad_weight = grad_bias = None

        current_stream = paddle.device.cuda.current_stream()
        # print("———————--------OpParallelLinear backward----------———————2")
        if (not input.stop_gradient) or (not weight.stop_gradient):
            if ctx.gather_input:
                # async the all_gather
                with paddle.device.cuda.stream_guard(config["tp_comm_stream"]):
                    input.record_stream(config["tp_comm_stream"])
                    config["tp_comm_stream"].wait_stream(current_stream)
                    all_input = preprocess_input(
                        input, ctx.gather_input, ctx.split_input
                    )
                    # use event to solve two streams waiting for each other
                    gather_event = config["tp_comm_stream"].record_event()
            else:
                all_input = preprocess_input(input, ctx.gather_input, ctx.split_input)

        # print("———————--------------OpParallelLinear backward----------———————3")
        # print("----------OpParallelLinear backward----------", input.shape,
        #       grad_output.shape, weight.shape)
        if not input.stop_gradient:
            # print("before grad_output.matmul(weight)", grad_output, weight)
            grad_all_input = grad_output.matmul(weight.T)
            # print("after grad_output.matmul(weight)", grad_all_input)
            grad_input = paddle.zeros_like(input)
            if ctx.gather_input:
                # async the reduce_scatter
                with paddle.device.cuda.stream_guard(config["tp_comm_stream"]):
                    config["tp_comm_stream"].wait_stream(current_stream)
                    # grad_input.record_stream(config["tp_comm_stream"])
                    # grad_all_input.record_stream(config["tp_comm_stream"])
                    nccl.reduceScatter(
                        grad_all_input,
                        grad_input,
                        "sum",
                        config["tp_comm"],
                    )
                # print("after nccl.reduceScatter(", grad_input)
            elif ctx.reduce_output_type is None:
                with paddle.device.cuda.stream_guard(config["tp_comm_stream"]):
                    config["tp_comm_stream"].wait_stream(current_stream)
                    # grad_input.record_stream(config["tp_comm_stream"])
                    nccl.allReduce(
                        grad_all_input,
                        grad_all_input,
                        "sum",
                        config["tp_comm"],
                    )
                    grad_input = grad_all_input
            else:
                grad_input = grad_all_input

            if ctx.split_input:
                with paddle.device.cuda.stream_guard(config["tp_comm_stream"]):
                    config["tp_comm_stream"].wait_stream(current_stream)
                    # grad_input.record_stream(config["tp_comm_stream"])
                    grad_input = all_gather(grad_input, config["tp_comm"])

        # print("———————--------------OpParallelLinear backward----------———————4")
        # wait all_gather
        if ctx.gather_input:
            current_stream.wait_event(gather_event)
        if not weight.stop_gradient:
            grad_weight = (
                grad_output.reshape([-1, grad_output.shape[-1]])
                .t()
                .matmul(all_input.reshape([-1, all_input.shape[-1]]))
            )
        # print("———————--------------OpParallelLinear backward----------———————5")
        if bias is not None and not bias.stop_gradient:
            grad_bias = grad_output.reshape([-1, grad_output.shape[-1]]).sum(0)

        current_stream = paddle.device.cuda.current_stream()
        current_stream.wait_stream(config["tp_comm_stream"])

        # print("------------OpParallelLinear return 输入", input, weight, bias)
        # print("------------OpParallelLinear return",
        #       grad_input, grad_weight, grad_bias, None, None, None, None, None)
        # return grad_input, grad_weight, grad_bias, None, None, None, None, None
        print("OpParallelLinear return---------------- forward保存的内容", input, weight, bias)

        print("OpParallelLinear return---------------- 计算出来的梯度", grad_input, grad_weight, grad_bias)
        returns = []
        if not input.stop_gradient:
            returns.append(grad_input)
        if not weight.stop_gradient:
            returns.append(grad_weight)
        if (not bias is None) and (not bias.stop_gradient):
            returns.append(grad_bias)
        
        return tuple(returns)
        # if bias is not None and not bias.stop_gradient:
        #     print("[Backward] 返回3个梯度 (input, weight, bias)")
        #     return grad_input, grad_weight, grad_bias
        # else:
        #     print("[Backward] 返回2个梯度 (input, weight)")
        #     return (grad_input, grad_weight)
    @staticmethod
    def backward_new(ctx, grad_output):
        print(f"[Backward] 开始，梯度输出形状: {grad_output.shape}")
        
        # 获取前向保存的输入
        input, weight, bias = ctx.saved_tensor()
        print(f"[保存张量] 输入形状: {input.shape if input is not None else None}")
        print(f"[保存张量] 权重形状: {weight.shape}")
        print(f"[保存张量] 偏置: {'存在' if bias is not None else '无'}")
        
        # 梯度初始化
        grad_input = None
        grad_weight = None
        grad_bias = None
        
        # 仅处理需要梯度的参数
        if not input.stop_gradient:
            print("[计算输入梯度] 开始...")
            # 简化: 直接使用矩阵乘法计算输入梯度
            try:
                print(f"[输入梯度] 计算开始, 形状: {grad_output.shape}, {weight.shape} ")
                grad_input = grad_output @ weight.T
                print(f"[输入梯度] 计算完成, 形状: {grad_input.shape}, 范数: {paddle.linalg.norm(grad_input).item():.4f}")
            except Exception as e:
                print(f"[输入梯度] 计算失败: {e}")
        
        if not weight.stop_gradient:
            print("[计算权重梯度] 开始...")
            try:
                # 简化: 直接使用矩阵乘法计算权重梯度
                grad_weight = grad_output.reshape([-1, grad_output.shape[-1]]).t() @ input.reshape([-1, input.shape[-1]])
                print(f"[权重梯度] 计算完成, 形状: {grad_weight.shape}, 范数: {paddle.linalg.norm(grad_weight).item():.4f}")
            except Exception as e:
                print(f"[权重梯度] 计算失败: {e}")
        
        if bias is not None and not bias.stop_gradient:
            print("[计算偏置梯度] 开始...")
            try:
                # 简化: 直接沿批量维度求和
                grad_bias = grad_output.reshape([-1, grad_output.shape[-1]]).sum(0)
                print(f"[偏置梯度] 计算完成, 形状: {grad_bias.shape}, 范数: {paddle.linalg.norm(grad_bias).item():.4f}")
            except Exception as e:
                print(f"[偏置梯度] 计算失败: {e}")
        
        # 同步CUDA流确保计算完成
        paddle.device.cuda.synchronize()
        print("[Backward] 所有梯度计算完成，准备返回")
        
        # 调试信息: 检查梯度值范围
        self_check = []
        if grad_input is not None:
            self_check.append(("grad_input", grad_input))
        if grad_weight is not None:
            self_check.append(("grad_weight", grad_weight))
        if grad_bias is not None:
            self_check.append(("grad_bias", grad_bias))
        
        for name, grad in self_check:
            if grad is None:
                continue
            min_val = grad.min().item()
            max_val = grad.max().item()
            mean_val = grad.mean().item()
            print(f"[梯度检查] {name}: min={min_val:.6f}, max={max_val:.6f}, mean={mean_val:.6f}")
            
            # 检查NaN/Inf
            if paddle.isnan(grad).any():
                print(f"!! 警告: {name} 包含NaN值")
            if paddle.isinf(grad).any():
                print(f"!! 警告: {name} 包含Inf值")
        
        # 返回简化结果
        print("[Backward] 返回梯度")
        if bias is not None:
            return grad_input, grad_weight, grad_bias
        else:
            return grad_input, grad_weight