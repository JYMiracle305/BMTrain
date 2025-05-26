from .. import C
import paddle

from ..utils import tensor_to_c_ptr

CHECK_INPUT = lambda x: x.is_contiguous() and x.place.is_gpu_place()


def has_inf_nan(g_half: paddle.Tensor, out: paddle.Tensor) -> None:
    assert out.dtype == paddle.uint8, "out must be a uint8 tensor"
    # print("g_half.place, out.place :::", g_half.place, out.place)
    assert CHECK_INPUT(g_half), "g_fp16 must be contiguous and on cuda"
    assert CHECK_INPUT(out), "out must be contiguous and on cuda"
    mid = paddle.zeros(1024, dtype=out.dtype)
    if out.place.is_gpu_place():
        mid = mid.cuda()
    stream = paddle.device.cuda.current_stream().cuda_stream
    if g_half.dtype == paddle.float16:
        # print(f"has_inf_nan 指针 {g_half.numel().item()}, {hex(tensor_to_c_ptr(g_half))}, \
        #       {hex(tensor_to_c_ptr(mid))}, {hex(tensor_to_c_ptr(out))}, {stream}")
        C.has_nan_inf_fp16_launcher(
            g_half.numel().item(), tensor_to_c_ptr(g_half), tensor_to_c_ptr(mid), tensor_to_c_ptr(out), stream
        )
    elif g_half.dtype == paddle.bfloat16:
        if not C.is_bf16_supported():
            raise NotImplementedError(f"bfloat16 is not supported on current GPU")
        C.has_nan_inf_bf16_launcher(
            g_half.numel().item(), tensor_to_c_ptr(g_half), tensor_to_c_ptr(mid), tensor_to_c_ptr(out), stream
        )
    else:
        raise ValueError(f"has_inf_nan not supported for dtype {g_half.dtype}")


def cross_entropy_forward(
    m: int,
    n: int,
    input: paddle.Tensor,
    target: paddle.Tensor,
    softmax: paddle.Tensor,
    output: paddle.Tensor,
    ignore_index: int,
) -> None:
    CHECK_INPUT(input)
    CHECK_INPUT(target)
    CHECK_INPUT(softmax)
    CHECK_INPUT(output)
    # print("cross_entropy_forward", target.dtype, output.dtype)
    assert target.dtype == paddle.int32, "target must be an int tensor"
    assert output.dtype == paddle.float32, "output must be a float tensor"
    assert (
        input.numel().item() == softmax.numel().item()
    ), "input and softmax must have the same number of elements"
    assert (
        target.numel().item() == output.numel().item()
    ), "target and output must have the same number of elements"
    input_ptr = tensor_to_c_ptr(input)
    target_ptr = tensor_to_c_ptr(target)
    softmax_ptr = tensor_to_c_ptr(softmax)
    output_ptr = tensor_to_c_ptr(output)
    cuda_stream = paddle.device.cuda.current_stream().cuda_stream
    if input.dtype == paddle.float16:
        C.cross_entropy_forward_fp16_launcher(
            m,
            n,
            input_ptr,
            target_ptr,
            softmax_ptr,
            output_ptr,
            ignore_index,
            cuda_stream,
        )
    elif input.dtype == paddle.bfloat16:
        if not C.is_bf16_supported():
            raise NotImplementedError(f"bfloat16 is not supported on current GPU")
        C.cross_entropy_forward_bf16_launcher(
            m,
            n,
            input_ptr,
            target_ptr,
            softmax_ptr,
            output_ptr,
            ignore_index,
            cuda_stream,
        )
    else:
        raise ValueError(f"cross_entropy_forward not supported for dtype {input.dtype}")


def cross_entropy_backward_inplace(
    m: int,
    n: int,
    grad_output: paddle.Tensor,
    target: paddle.Tensor,
    x: paddle.Tensor,
    ignore_index: int,
) -> None:
    CHECK_INPUT(grad_output)
    CHECK_INPUT(target)
    CHECK_INPUT(x)
    assert grad_output.dtype == paddle.float32, "grad_output must be a float tensor"
    assert target.dtype == paddle.int32, "target must be an int tensor"
    assert (
        target.numel().item() == grad_output.numel().item()
    ), "target and grad_output must have the same number of elements"
    cuda_stream = paddle.device.cuda.current_stream().cuda_stream
    grad_output_ptr = tensor_to_c_ptr(grad_output)
    target_ptr = tensor_to_c_ptr(target)
    x_ptr = tensor_to_c_ptr(x)

    if x.dtype == paddle.float16:
        C.cross_entropy_backward_inplace_fp16_launcher(
            m, n, grad_output_ptr, target_ptr, x_ptr, ignore_index, cuda_stream
        )
    elif x.dtype == paddle.bfloat16:
        if not C.is_bf16_supported():
            raise NotImplementedError(f"bfloat16 is not supported on current GPU")
        C.cross_entropy_backward_inplace_bf16_launcher(
            m, n, grad_output_ptr, target_ptr, x_ptr, ignore_index, cuda_stream
        )
    else:
        raise ValueError(
            f"cross_entropy_backward not supported for dtype {input.dtype}"
        )


def fused_sumexp(logits: paddle.Tensor, max_logits: paddle.Tensor) -> paddle.Tensor:
    CHECK_INPUT(logits)
    CHECK_INPUT(max_logits)
    assert max_logits.dtype == paddle.float32, "max_logits must be float tensor"
    assert max_logits.shape[0] == logits.shape[0], "max_logits must have same shape[0] as logits"
    sum_exp_logits = paddle.empty([logits.shape[0]], dtype = paddle.float32)
    if logits.place.is_gpu_place():
        sum_exp_logits = sum_exp_logits.cuda()
    m = logits.shape[0]
    n = logits.shape[1]
    cuda_stream = paddle.device.cuda.current_stream().cuda_stream
    logits_ptr = tensor_to_c_ptr(logits)
    max_logits_ptr = tensor_to_c_ptr(max_logits)
    sum_exp_logits_ptr = tensor_to_c_ptr(sum_exp_logits)
    if logits.dtype == paddle.float16:
        C.fused_sumexp_fp16_launcher(
            m, n, logits_ptr, max_logits_ptr, sum_exp_logits_ptr, cuda_stream
        )
    elif logits.dtype == paddle.bfloat16:
        if not C.is_bf16_supported():
            raise NotImplementedError(f"bfloat16 is not supported on current GPU")
        C.fused_sumexp_bf16_launcher(
            m, n, logits_ptr, max_logits_ptr, sum_exp_logits_ptr, cuda_stream
        )
    else:
        raise ValueError(f"fused_sumexp not supported for dtype {logits.dtype}")
    return sum_exp_logits


def fused_softmax_inplace(
    logits: paddle.Tensor, max_logits: paddle.Tensor, sum_exp_logits: paddle.Tensor
) -> None:
    CHECK_INPUT(logits)
    CHECK_INPUT(max_logits)
    CHECK_INPUT(sum_exp_logits)
    assert max_logits.dtype == paddle.float32, "max_logits must be float tensor"
    assert sum_exp_logits.dtype == paddle.float32, "sum_exp_logits must be float tensor"
    assert max_logits.shape[0] == logits.shape[0], "max_logits must have same shape[0] as logits"
    assert sum_exp_logits.shape[0] == logits.shape[0], "sum_exp_logits must have same shape[0] as logits"
    m = logits.shape[0]
    n = logits.shape[1]
    cuda_stream = paddle.device.cuda.current_stream().cuda_stream
    logits_ptr = tensor_to_c_ptr(logits)
    max_logits_ptr = tensor_to_c_ptr(max_logits)
    sum_exp_logits_ptr = tensor_to_c_ptr(sum_exp_logits)
    if logits.dtype == paddle.float16:
        C.fused_softmax_inplace_fp16_launcher(
            m, n, logits_ptr, max_logits_ptr, sum_exp_logits_ptr, cuda_stream
        )
    elif logits.dtype == paddle.bfloat16:
        if not C.is_bf16_supported():
            raise NotImplementedError(f"bfloat16 is not supported on current GPU")
        C.fused_softmax_inplace_bf16_launcher(
            m, n, logits_ptr, max_logits_ptr, sum_exp_logits_ptr, cuda_stream
        )
    else:
        raise ValueError(
            f"fused_softmax_inplace not supported for dtype {logits.dtype}"
        )
