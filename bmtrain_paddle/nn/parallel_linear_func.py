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
    print("---------------async_all_gather_linear_func before", input.shape, weight.shape, bias.shape)
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
    print("--------------- 11111111111 async_all_gather_linear_func", input)
    input = all_gather(inputs[0], config["tp_comm"])
    print("--------------- 22222222222 async_all_gather_linear_func", input.shape)

    input = input.flatten(0, 1)
    print("--------------- 33333333333 async_all_gather_linear_func", input, weight)
    # 检查输入和权重的数值范围
    print("input max/min:", input.max().item(), input.min().item())
    print("weight max/min:", weight.max().item(), weight.min().item())

    out = F.linear(input, weight, bias)
    print("--------------- 44444444444 async_all_gather_linear_func", out)
    outs = out.chunk(tp_size, axis=0)
    print("---------------async_all_gather_linear_func after after", input, outs)
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
        print("---------------async_all_gather_linear_func after after !!!!!!!", input, outputs)
    out = paddle.concat(outputs, axis=0)
    if dim > 2:
        out_shape = list(out.shape)
        shape[-1] = out_shape[-1]
        shape[0] = shape[0] * tp_size
        out = out.reshape(shape)

    print("---------------async_all_gather_linear_func res", out)

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
            print("async_reduce_scatter_linear_func shape[0]", shape[0])
            outputs[i] = paddle.empty(shape, dtype=out.dtype)
            print("async_reduce_scatter_linear_func out outputs[0]", out.shape, outputs[i].shape,
                  out.numel().item(), outputs[i].numel().item())
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
        if reduce_output_type is not None:
            reduce_output_type = ReduceType(reduce_output_type)
        print("OpParallelLinear forward", input.shape, weight.shape, bias)
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
            print(f"!!!!!!!!!!!!!!!!!!!!! input, weight, ctx.split_input {all_input.shape} {weight.shape} {bias}")
            out = F.linear(all_input, weight, bias)
            print(f"OpParallelLinear F.linear {out.shape}")
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

        if ctx.reduce_output_type == ReduceType.REDUCE_SCATTER:
            print("———————--------OpParallelLinear backward------------———————1")
            if (not input.stop_gradient) or (not weight.stop_gradient):
                grad_input, grad_weight, grad_bias = (
                    async_all_gather_linear_backward_func(
                        grad_output, input, weight, bias, ctx.async_gather_chunks
                    )
                )
                return grad_input, grad_weight, grad_bias, None, None, None, None, None
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
        print("———————--------OpParallelLinear backward----------———————2")
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

        print("———————--------------OpParallelLinear backward----------———————3")
        print("----------OpParallelLinear backward----------", input.shape, grad_output.shape, weight.shape)
        if not input.stop_gradient:
            grad_all_input = grad_output.matmul(weight)
            grad_input = paddle.zeros_like(input)
            if ctx.gather_input:
                # async the reduce_scatter
                with paddle.device.cuda.stream_guard(config["tp_comm_stream"]):
                    config["tp_comm_stream"].wait_stream(current_stream)
                    grad_input.record_stream(config["tp_comm_stream"])
                    grad_all_input.record_stream(config["tp_comm_stream"])
                    nccl.reduceScatter(
                        grad_all_input,
                        grad_input,
                        "sum",
                        config["tp_comm"],
                    )
            elif ctx.reduce_output_type is None:
                with paddle.device.cuda.stream_guard(config["tp_comm_stream"]):
                    config["tp_comm_stream"].wait_stream(current_stream)
                    grad_input.record_stream(config["tp_comm_stream"])
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
                    grad_input.record_stream(config["tp_comm_stream"])
                    grad_input = all_gather(grad_input, config["tp_comm"])

        print("———————--------------OpParallelLinear backward----------———————4")
        # wait all_gather
        if ctx.gather_input:
            current_stream.wait_event(gather_event)
        if not weight.stop_gradient:
            grad_weight = (
                grad_output.reshape([-1, grad_output.shape[-1]])
                .t()
                .matmul(all_input.reshape([-1, all_input.shape[-1]]))
            )
        print("———————--------------OpParallelLinear backward----------———————5")
        if bias is not None and not bias.stop_gradient:
            grad_bias = grad_output.reshape([-1, grad_output.shape[-1]]).sum(0)

        current_stream = paddle.device.cuda.current_stream()
        current_stream.wait_stream(config["tp_comm_stream"])
        return grad_input, grad_weight, grad_bias, None, None, None, None, None
