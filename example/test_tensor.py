import torch
import paddle
import ctypes
from typing import Any, Dict, Iterable, Optional, Tuple
import cupy

def tensor_to_c_ptr(tensor: paddle.Tensor):
    if not tensor.place.is_gpu_place():
        tensor = tensor.cuda()
    tensor = tensor.contiguous()
    
    # 通过 DLPack 获取 CuPy 数组（共享内存）
    dlpack = tensor.value().get_tensor()._to_dlpack()
    cp_array = cupy.fromDlpack(dlpack)

    return cp_array.data.ptr

input = torch.empty(2, 3, dtype=torch.float32, device="cuda")

print("torch", input.storage().size(), input.numel())
print("storage_type()", input.storage_type())

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


c_ptr = tensor_to_c_ptr(input_paddle_cuda)
print("指针", input_paddle_cuda.place, c_ptr)

