import paddle
import numpy as np
import bmtrain_paddle as bmt
from bmtrain_paddle import init_distributed
import os
import fcntl
import time
import shutil
import cupy
import ctypes
from weakref import WeakKeyDictionary
# 自定义的 allReduce 接口
from bmtrain_paddle.global_var import config

class FileStore:
    def __init__(self, prefix="bmtrain_store"):
        # 唯一路径隔离不同训练任务
        self.base_path = f"./tmp/{prefix}_{os.getenv('MASTER_ADDR')}_{os.getenv('MASTER_PORT')}"

        # if dist.get_rank() == 0:
        #     self._cleanup_old_directory()

        # 清理旧目录并创建新目录
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
        print(f"FileStore path: {self.base_path}")
    def set(self, key: str, value: str):
        """原子写入键值对，避免读写冲突"""
        file_path = os.path.join(self.base_path, key)
        tmp_path = f"{file_path}.tmp"
        
        # 写入临时文件后原子重命名
        with open(tmp_path, "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX)  # 排他锁
            print("FileStore set", tmp_path, value)
            f.write(value)
            f.flush()
            os.fsync(f.fileno())  # 强制刷盘
            fcntl.flock(f, fcntl.LOCK_UN)
        
        os.rename(tmp_path, file_path)  # Unix 原子操作

    def get(self, key: str, retries=30, interval=0.1) -> str:
        """安全读取键值，支持重试避免竞争"""
        file_path = os.path.join(self.base_path, key)
        print("FileStore get", file_path)
        for _ in range(retries):
            if os.path.exists(file_path):
                try:
                    with open(file_path, "r") as f:
                        fcntl.flock(f, fcntl.LOCK_SH)  # 共享锁
                        content = f.read()
                        print("FileStore get", file_path, content)
                        fcntl.flock(f, fcntl.LOCK_UN)
                        return content
                except FileNotFoundError:
                    pass  # 文件在检查后可能被删除
            time.sleep(interval)
        raise TimeoutError(f"Key {key} not found after {retries * interval}s")

    def delete(self, key: str):
        """删除键值"""
        file_path = os.path.join(self.base_path, key)
        if os.path.exists(file_path):
            os.remove(file_path)

_tensor_refs = WeakKeyDictionary()

def tensor_to_c_ptr(tensor: paddle.Tensor):
    if not isinstance(tensor, paddle.Tensor):
        raise TypeError("输入必须为 paddle.Tensor 类型")
    
    # 确保张量在 GPU 上且内存连续
    if not tensor.place.is_gpu_place():
        raise RuntimeError("仅支持 GPU Tensor")
    tensor = tensor.contiguous()

    # 将 Tensor 转换为 CuPy 数组，并维护引用
    cupy_array = cupy.asarray(tensor)
    # _tensor_refs[tensor] = cupy_array  # 使用弱引用字典

    # 获取指针并返回元信息
    return ctypes.c_void_p(cupy_array.data.ptr)

def test_allreduce():
    # os.environ["PADDLE_MASTER"] = f"localhost:12345"
    # os.environ["PADDLE_TRAINER_ENDPOINTS"] = f"localhost:12345,localhost:12346"

    # 初始化分布式环境（假设使用单机多卡）
    paddle.distributed.init_parallel_env()
    
    # 打印进程信息
    rank = paddle.distributed.get_rank()
    world_size = paddle.distributed.get_world_size()
    
    if os.path.exists("./tmp"):
        shutil.rmtree("./tmp")
    store = FileStore()
    # 创建 NCCL 通信子
    if rank == 0:
        unique_id: bytes = bmt.nccl.getUniqueId().hex()
        # TODO 进程间通信：设置一个唯一的id，一个进程设置，其他进程可以获取到nccl
        store.set("BMTRAIN_UNIQUE_ID", unique_id)
        print(f"rank {rank} unique_id {unique_id}")
    else:
        unique_id = store.get("BMTRAIN_UNIQUE_ID")
        print(f"rank {rank} unique_id {unique_id}")

    print("-----------------bytes.fromhex(unique_id), world_size, rank", bytes.fromhex(unique_id), world_size, rank)
    comm = bmt.nccl.commInitRank(bytes.fromhex(unique_id), world_size, rank)

    # 创建测试张量（相同 shape/dtype，在 GPU 上）
    rank = paddle.distributed.get_rank()
    src = paddle.to_tensor([rank + 1], dtype=paddle.float32).cuda()  # 示例数据：不同 rank 的值不同
    dst = paddle.zeros(shape=src.shape, dtype=src.dtype).cuda()  # 目标张量

    input_paddle = paddle.empty(shape=[4, 9], dtype=paddle.float32).cuda()

    sendbuff = tensor_to_c_ptr(src)
    recvbuff = tensor_to_c_ptr(dst)
    input_paddle_ptr = tensor_to_c_ptr(input_paddle)
    print(f"src_ptr: {hex(sendbuff.value)}")  # 应为非零地址（如0x7f8a5c000000）
    print(f"dst_ptr: {hex(recvbuff.value)}")
    print(f"dst_ptr: {recvbuff}")
    print(f"input_paddle_ptr: {hex(input_paddle_ptr.value)}")
    print(f"input_paddle_ptr: {input_paddle_ptr}")

    print("src dst", src.place, dst.place)
    
    # 打印初始值
    print(f"Rank {rank} | Before AllReduce:")
    print(f"src: {src.numpy()}, dst: {dst.numpy()}")
    print(f"dst.place {dst.place}")

    device_id = dst.place.gpu_device_id()
    paddle.device.cuda.synchronize(device_id)

    # 执行 AllReduce（示例使用 sum 操作）
    bmt.nccl.allReduce(
        src=src,
        dst=dst,
        op="sum",
        comm=comm
    )

    # 等待操作完成
    paddle.device.cuda.synchronize(device_id)

    # 打印结果
    print(f"Rank {rank} | After AllReduce (sum):")
    print(f"src: {src.numpy()}, dst: {dst.numpy()}")

    # 验证结果
    expected_sum = sum(range(1, paddle.distributed.get_world_size() + 1))  # 1+2+...+world_size
    assert np.allclose(dst.numpy(), [expected_sum]), "AllReduce Result Mismatch!"

if __name__ == "__main__":
    test_allreduce()
    # paddle.distributed.spawn(test_allreduce, nprocs=2)