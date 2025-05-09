from .. import C
import paddle
from ..utils import tensor_to_c_ptr

CHECK_INPUT = lambda x: x.is_contiguous() and x.place.is_gpu_place()


def bf16_from_fp32(param_fp32):
    param_bf16 = paddle.empty_like(param_fp32, dtype=paddle.bfloat16)
    C.to_bf16_from_fp32(
        param_fp32.size, tensor_to_c_ptr(param_fp32), tensor_to_c_ptr(param_bf16)
    )
    return param_bf16


def fp16_from_fp32(param_fp32):
    param_fp16 = paddle.empty_like(param_fp32, dtype=paddle.float16)
    C.to_fp16_from_fp32(
        param_fp32.size, tensor_to_c_ptr(param_fp32), tensor_to_c_ptr(param_fp16)
    )
    return param_fp16


def adam_cpu(
    param_fp32: paddle.Tensor,
    param_fp16: paddle.Tensor,
    delta_info: paddle.Tensor,
    g_fp16: paddle.Tensor,
    m_fp32: paddle.Tensor,
    v_fp32: paddle.Tensor,
    beta1: float,
    beta2: float,
    eps: float,
    lr: float,
    scale: float,
    weight_decay: float,
    step: int,
) -> None:
    assert param_fp32.is_contiguous(), "param_fp32 must be contiguous"
    assert param_fp16.is_contiguous(), "param_fp16 must be contiguous"
    assert g_fp16.is_contiguous(), "g_fp16 must be contiguous"
    assert m_fp32.is_contiguous(), "m_fp32 must be contiguous"
    assert v_fp32.is_contiguous(), "v_fp32 must be contiguous"
    assert param_fp32.dtype == paddle.float32, "param_fp32 must be float32 tensor"
    assert (
        param_fp16.dtype == paddle.float16 or param_fp16.dtype == paddle.bfloat16
    ), "param_fp16 must be float16/bfloat16 tensor"
    assert (
        g_fp16.dtype == paddle.float16 or g_fp16.dtype == paddle.bfloat16
    ), "g_fp16 must be float16/bfloat16 tensor"
    assert m_fp32.dtype == paddle.float32, "m_fp32 must be float32 tensor"
    assert v_fp32.dtype == paddle.float32, "v_fp32 must be float32 tensor"
    print("adam_cpu", param_fp32.place, param_fp32.place.is_cpu_place)
    assert param_fp32.place.is_cpu_place, "param_fp32 must be a cpu tensor"
    assert param_fp16.place.is_cpu_place, "param_fp16 must be a cpu tensor"
    assert g_fp16.place.is_cpu_place, "g_fp16 must be a cpu tensor"
    assert m_fp32.place.is_cpu_place, "m_fp32 must be a cpu tensor"
    assert v_fp32.place.is_cpu_place, "v_fp32 must be a cpu tensor"
    assert (
        param_fp32.size == param_fp16.size
    ), "param_fp32 and param_fp16 must have the same number of elements"
    assert (
        param_fp32.size == g_fp16.size
    ), "param_fp32 and g_fp16 must have the same number of elements"
    assert (
        param_fp32.size == m_fp32.size
    ), "param_fp32 and m_fp32 must have the same number of elements"
    assert (
        param_fp32.size == v_fp32.size
    ), "param_fp32 and v_fp32 must have the same number of elements"
    if delta_info is not None:
        assert delta_info.is_contiguous(), "delta_info must be contiguous"
        assert delta_info.dtype == paddle.float32, "delta_info must be float32 tensor"
        assert isinstance(delta_info.place, paddle.CPUPlace), "delta_info must be a cpu tensor"
        assert delta_info.size == 4, "delta_info have a length of 4"
    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step
    if g_fp16.dtype == paddle.float16:
        print("g_fp16.dtype == paddle.float16")
        launcher = C.adam_cpu_fp16_launcher
    elif g_fp16.dtype == paddle.bfloat16:
        print("g_fp16.dtype == paddle.bfloat16")
        if not C.is_bf16_supported():
            raise NotImplementedError(f"bfloat16 is not supported on current GPU")
        launcher = C.adam_cpu_bf16_launcher
    print("adam_cpu 指针", tensor_to_c_ptr(param_fp32),
        hex(tensor_to_c_ptr(param_fp32)),
        hex(tensor_to_c_ptr(param_fp16)),
        hex(tensor_to_c_ptr(delta_info) if delta_info is not None else 0),
        hex(tensor_to_c_ptr(g_fp16)),
        hex(tensor_to_c_ptr(m_fp32)),
        hex(tensor_to_c_ptr(v_fp32)),
    )
    launcher(
        tensor_to_c_ptr(param_fp32),
        tensor_to_c_ptr(param_fp32),
        tensor_to_c_ptr(param_fp16),
        tensor_to_c_ptr(delta_info) if delta_info is not None else 0,
        tensor_to_c_ptr(g_fp16),
        tensor_to_c_ptr(m_fp32),
        tensor_to_c_ptr(v_fp32),
        beta1,
        beta2,
        eps,
        lr,
        scale,
        weight_decay,
        bias_correction1,
        bias_correction2,
    )


def adam_fp16(
    param_fp32: paddle.Tensor,
    param_fp16: paddle.Tensor,
    g_fp16: paddle.Tensor,
    m_fp16: paddle.Tensor,
    v_fp32: paddle.Tensor,
    beta1: float,
    beta2: float,
    eps: float,
    lr: float,
    scale: float,
    weight_decay: float,
    step: int,
) -> None:
    assert CHECK_INPUT(param_fp32), "param_fp32 must be contiguous and on cuda"
    assert CHECK_INPUT(param_fp16), "param_fp16 must be contiguous and on cuda"
    assert CHECK_INPUT(g_fp16), "g_fp16 must be contiguous and on cuda"
    assert CHECK_INPUT(m_fp16), "m_fp32 must be contiguous and on cuda"
    assert CHECK_INPUT(v_fp32), "v_fp32 must be contiguous and on cuda"
    assert param_fp32.dtype == paddle.float32, "param_fp32 must be float32 tensor"
    assert param_fp16.dtype == paddle.float16, "param_fp16 must be float16 tensor"
    assert g_fp16.dtype == paddle.float16, "g_fp16 must be float16 tensor"
    assert m_fp16.dtype == paddle.float16, "m_fp16 must be float16 tensor"
    assert v_fp32.dtype == paddle.float32, "v_fp32 must be float32 tensor"
    assert (
        param_fp32.size == param_fp16.size
    ), "param_fp32 and param_fp16 must have the same number of elements"
    assert (
        param_fp32.size == g_fp16.size
    ), "param_fp32 and g_fp16 must have the same number of elements"
    assert (
        param_fp32.size == m_fp16.size
    ), "param_fp32 and m_fp32 must have the same number of elements"
    assert (
        param_fp32.size == v_fp32.size
    ), "param_fp32 and v_fp32 must have the same number of elements"
    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step
    stream = paddle.device.cuda.current_stream().cuda_stream
    C.adam_fp16_launcher(
        int(param_fp32.size),
        tensor_to_c_ptr(param_fp32),
        tensor_to_c_ptr(param_fp16),
        tensor_to_c_ptr(g_fp16),
        tensor_to_c_ptr(m_fp16),
        tensor_to_c_ptr(v_fp32),
        beta1,
        beta2,
        eps,
        lr,
        scale,
        weight_decay,
        bias_correction1,
        bias_correction2,
        stream,
    )


def adam_bf16(
    param_fp32: paddle.Tensor,
    param_bf16: paddle.Tensor,
    g_bf16: paddle.Tensor,
    m_fp32: paddle.Tensor,
    v_fp32: paddle.Tensor,
    beta1: float,
    beta2: float,
    eps: float,
    lr: float,
    scale: float,
    weight_decay: float,
    step: int,
) -> None:
    assert CHECK_INPUT(param_fp32), "param_fp32 must be contiguous and on cuda"
    assert CHECK_INPUT(param_bf16), "param_bf16 must be contiguous and on cuda"
    assert CHECK_INPUT(g_bf16), "g_bf16 must be contiguous and on cuda"
    assert CHECK_INPUT(m_fp32), "m_fp32 must be contiguous and on cuda"
    assert CHECK_INPUT(v_fp32), "v_fp32 must be contiguous and on cuda"
    assert param_fp32.dtype == paddle.float32, "param_fp32 must be float32 tensor"
    assert param_bf16.dtype == paddle.bfloat16, "param_fp16 must be float16 tensor"
    assert g_bf16.dtype == paddle.bfloat16, "g_bf16 must be bfloat16 tensor"
    assert m_fp32.dtype == paddle.float32, "m_fp32 must be bfloat16 tensor"
    assert v_fp32.dtype == paddle.float32, "v_fp32 must be float32 tensor"
    assert (
        param_fp32.size == param_bf16.size
    ), "param_fp32 and param_bf16 must have the same number of elements"
    assert (
        param_fp32.size == g_bf16.size
    ), "param_fp32 and g_fp16 must have the same number of elements"
    assert (
        param_fp32.size == m_fp32.size
    ), "param_fp32 and m_m_fp32 must have the same number of elements"
    assert (
        param_fp32.size == v_fp32.size
    ), "param_fp32 and v_fp32 must have the same number of elements"
    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step
    stream = paddle.device.cuda.current_stream().cuda_stream
    if not C.is_bf16_supported():
        raise NotImplementedError(f"bfloat16 is not supported on current GPU")
    C.adam_bf16_launcher(
        param_fp32.size,
        tensor_to_c_ptr(param_fp32),
        tensor_to_c_ptr(param_bf16),
        tensor_to_c_ptr(g_bf16),
        tensor_to_c_ptr(m_fp32),
        tensor_to_c_ptr(v_fp32),
        beta1,
        beta2,
        eps,
        lr,
        scale,
        weight_decay,
        bias_correction1,
        bias_correction2,
        stream,
    )
