import torch
import paddle
import ctypes
from typing import Any, Dict, Iterable, Optional, Tuple

def tensor_to_c_ptr(tensor: paddle.Tensor) -> Tuple[ctypes.POINTER(ctypes.c_void_p), Tuple[int], Dict[str, Any]]:
    """
    将 PaddlePaddle Tensor 转换为 C 指针及元信息（适用于 CPU/GPU Tensor）。

    Args:
        tensor (paddle.Tensor): 输入的 Paddle Tensor，支持任意形状和设备。

    Returns:
        Tuple:
            - c_ptr (ctypes.POINTER): 数据指针（void* 类型，需根据 dtype 转换为具体类型）。
            - shape (Tuple[int]): 张量形状。
            - meta (Dict): 包含 dtype（str）和 numel（int）的元信息。

    Raises:
        TypeError: 输入非 Tensor 时抛出异常。
    """
    if not isinstance(tensor, paddle.Tensor):
        raise TypeError("输入必须为 paddle.Tensor 类型")

    # 将 Tensor 移动到 CPU 并确保内存连续
    cpu_tensor = tensor.cpu().contiguous()

    # 转换为 NumPy 数组
    np_array = cpu_tensor.numpy()

    # 数据类型映射表（支持常见类型）
    dtype_map = {
        'float16': ctypes.c_float,
        'float32': ctypes.c_float,
        'int32': ctypes.c_int32,
        'int64': ctypes.c_int64,
        'uint8': ctypes.c_uint8,
    }

    dtype_str = str(cpu_tensor.dtype).split('.')[-1]
    if dtype_str not in dtype_map:
        raise ValueError(f"不支持的数据类型: {dtype_str}")

    # 获取指针、形状和元素数量
    c_type = dtype_map[dtype_str]
    c_ptr = np_array.ctypes.data_as(ctypes.POINTER(c_type))
    ptr_address = ctypes.cast(c_ptr, ctypes.c_void_p).value
    shape = tuple(cpu_tensor.shape)
    numel = cpu_tensor.numel().item()

    # 返回指针、形状及元数据
    return ptr_address, shape, {'dtype': dtype_str, 'numel': numel}

input = torch.empty(2, 3, dtype=torch.float32, device="cuda")

print("torch", input.storage().size(), input.numel())

input_paddle = paddle.empty(shape=[2, 3], dtype=paddle.float32)
input_paddle_2 = paddle.empty(shape=[2, 3], dtype=paddle.float32).cuda()
print("paddle", input_paddle.size, input_paddle.numel().item(), input_paddle.shape)

input_paddle_cuda = input_paddle
if (input_paddle_2.place.is_gpu_place()):
    input_paddle_cuda = input_paddle.cuda()
print("input_paddle_cuda", input_paddle_cuda.place)
raw = 5
shape_example = (1, 7, 8)

print((raw,) + shape_example)


c_ptr, shape, size = tensor_to_c_ptr(input_paddle_cuda)
print("指针", input_paddle_cuda.place, c_ptr, shape, size)
