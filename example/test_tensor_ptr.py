import paddle
import cupy as cp  # 需要安装 cupy

def tensor_to_c_ptr(tensor: paddle.Tensor):
    # 确保张量在 GPU 上且连续
    if not tensor.place.is_gpu_place():
        tensor = tensor.cuda()
    tensor = tensor.contiguous()
    
    # 通过 CuPy 共享 GPU 内存
    cp_array = cp.asarray(tensor)
    return cp_array.data.ptr, tensor.shape, tensor.dtype, cp_array


src = paddle.to_tensor([[1., 2., 3.], [4., 5., 6.]], dtype='float32').cuda()
ptr,_,_,cp_array = tensor_to_c_ptr(src)
print({hex(ptr)})

cp_array[0, 0] = 100.0
# print(src.type, cp_array.type)
print(src, cp_array)


import paddle
import cupy as cp

# def tensor_to_cupy(tensor: paddle.Tensor):
#     # 确保张量在 GPU 上且连续
#     if not tensor.place.is_gpu_place():
#         tensor = tensor.cuda()
#     tensor = tensor.contiguous()
    
#     # 通过 DLPack 协议转换共享内存
#     dlpack = tensor.value().get_tensor()._to_dlpack()  # 获取 DLPack 对象
#       # 零拷贝转换为 CuPy 数组
#     return dlpack

import paddle
import cupy as cp

def tensor_to_c_ptr(tensor: paddle.Tensor) -> tuple:
    # 确保张量在 GPU 上且连续
    if not tensor.place.is_gpu_place():
        tensor = tensor.cuda()
    tensor = tensor.contiguous()
    
    # 通过 DLPack 获取 CuPy 数组（共享内存）
    dlpack = tensor.value().get_tensor()._to_dlpack()
    cp_array = cp.fromDlpack(dlpack)
    
    # 返回指针、形状、数据类型
    return cp_array.data.ptr, tensor.shape, tensor.dtype

# 创建示例张量
src = paddle.to_tensor([[1., 2., 3.], [4., 5., 6.]], dtype='float32').cuda()

# 获取指针、形状、数据类型
ptr, shape, dtype = tensor_to_c_ptr(src)

print(ptr, shape, dtype)
# -------------------------------
# 通过指针构造 CuPy 数组并修改数据
# -------------------------------
print(dtype.name)
# 1. 将 Paddle dtype 转换为 CuPy dtype（例如 float32 → cp.float32）
# cp_dtype = getattr(cp, dtype.name.split('.')[-1])
cp_dtype = cp.float32
# 2. 计算内存总大小（元素数量 * 每个元素字节数）
element_size = cp.dtype(cp_dtype).itemsize
total_size = src.size * element_size

# 3. 构造 CuPy 数组（共享原张量的 GPU 内存）
mem = cp.cuda.UnownedMemory(ptr, total_size, owner=src)  # 绑定生命周期
mem_ptr = cp.cuda.MemoryPointer(mem, 0)
cp_array = cp.ndarray(shape, dtype=cp_dtype, memptr=mem_ptr)

# 4. 修改数据
cp_array[0, 0] = 99.0     # 修改第一个元素
cp_array[1, 2] = 88.0     # 修改最后一个元素

print("张量", src)
# 5. 打印原张量验证修改
print("修改后的张量:\n", src.numpy())