
from typing_extensions import Literal
import paddle
from .. import C
from .enums import *
import ctypes
from ..utils import tensor_to_c_ptr
class NCCLCommunicator:
    """
    NCCL communicator stores the communicator handle.
    """

    def __init__(self, ptr) -> None:
        self.__ptr = ptr
    
    @property
    def ptr(self):
        """
        Returns the communicator handle.
        """
        if self.__ptr == -1:
            raise RuntimeError("NCCL Communicator is already destroyed")
        return self.__ptr
    
    def _destroy_ptr(self):
        self.__ptr = -1

# utils

def dtype2nccl(dtype : paddle.dtype) -> int:
    MAP = {
        paddle.int8: ncclInt8,
        paddle.uint8: ncclUint8,
        paddle.int32: ncclInt32,
        paddle.int64: ncclInt64,
        paddle.float16: ncclFloat16,
        paddle.bfloat16: ncclBFloat16,
        paddle.float32: ncclFloat32,
        paddle.float64: ncclFloat64,
        paddle.bool: ncclBool
    }
    if dtype not in MAP:
        raise TypeError("Unsupport dtype %s" % dtype)
    return MAP[dtype]

def op2nccl(
    op : Literal["sum", "prod", "max", "min", "avg"]
):
    if op == "sum":
        return ncclSum
    if op == "prod":
        return ncclProd
    if op == "max":
        return ncclMax
    if op == "min":
        return ncclMin
    if op == "avg":
        return ncclAvg
    raise ValueError("Unknown gather op %s")

# wrappers

def getUniqueId() -> bytes:
    """
    NCCL API: `ncclGetUniqueId <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclgetuniqueid>`_

    """
    return C.ncclGetUniqueId()

def commInitRank(unique_id : bytes, world_size : int, rank : int) -> NCCLCommunicator:
    """
    NCCL API: `ncclCommInitRank <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclcomminitrank>`_

    """
    assert rank >= 0 and rank < world_size, "rank must be between 0 and world_size-1"
    return NCCLCommunicator(C.ncclCommInitRank(unique_id, world_size, rank))

def commDestroy(comm : NCCLCommunicator):
    """
    NCCL API: `ncclCommDestroy <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclcommdestroy>`_

    """
    C.ncclCommDestroy(comm.ptr)
    comm._destroy_ptr()
def commCount(comm : NCCLCommunicator):
    """NCCL API: `ncclCommCount <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclcommcount>`_

    Args:
        comm (NCCLCommunicator): NCCL communicator.
    """
    return C.ncclCommCount(comm.ptr)
### collective
def commRank(comm : NCCLCommunicator):
    """NCCL API: `ncclCommUserRank <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclCommUserRank>`_

    Args:
        comm (NCCLCommunicator): NCCL communicator.
    """
    return C.ncclCommUserRank(comm.ptr)

def allReduce(
        src : paddle.Tensor,
        dst : paddle.Tensor,
        op : Literal["sum", "prod", "max", "min", "avg"],
        comm : NCCLCommunicator
    ):
    """NCCL API: `ncclAllReduce <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#ncclallreduce>`_

    Args:
        src (torch.storage._StorageBase): Source buffer.
        dst (torch.storage._StorageBase): Destination buffer.
        op (Literal["sum", "prod", "max", "min", "avg"]): Reduction operation.
        comm (NCCLCommunicator): NCCL communicator.
    
    The src and dst buffers must be the same size, type and on the same device.

    If src == dst, the operation is performed in-place.

    """
    assert src.dtype == dst.dtype, "send and recv buffers must be the same type"
    assert src.place.is_gpu_place() and dst.place.is_gpu_place(), "Tensors must be on GPU"

    count = src.numel().item()
    datatype = dtype2nccl(src.dtype)
    operator = op2nccl(op)

    # print(f"src {src} dst {dst}")
    sendbuff = tensor_to_c_ptr(src)
    recvbuff = tensor_to_c_ptr(dst)

    # print(f"src_ptr: {hex(sendbuff)}")  # （如0x7f8a5c000000）
    # print(f"dst_ptr: {hex(recvbuff)}")  # 应为非零地址
    # print("src.shape == dst.shape", src.shape, dst.shape, src.size, dst.size)
    assert src.size == dst.size, "Buffer size not aligned"
    # print("------------nccl all reduce--------------", count,
    #       paddle.device.cuda.current_stream().cuda_stream)
    C.ncclAllReduce(
        sendbuff,
        recvbuff,
        count,
        datatype,
        operator,
        comm.ptr,
        paddle.device.cuda.current_stream().cuda_stream
    )
    # print(f"after src_ptr: {hex(sendbuff)}")  # （如0x7f8a5c000000）
    # print(f"after dst_ptr: {hex(recvbuff)}")  # 应为非零地址

def send(src : paddle.Tensor,
         peer : int,
         comm : NCCLCommunicator
    ):
    """NCCL API: `ncclsend <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/p2p.html#ncclsend>`_

        Args:
            src (torch.storage._StorageBase): Source buffer.
            peer (int): rank peer needs to call ncclRecv
            comm (NCCLCommunicator): NCCL communicator.
    """

    sendbuff = src._ptr()
    count = src.numel()
    datatype = dtype2nccl(src.dtype)
    C.ncclSend(
        sendbuff,
        count,
        datatype,
        peer,
        comm.ptr,
        paddle.device.cuda.current_stream().cuda_stream
    )
def recv(dst : paddle.Tensor,
         peer : int,
         comm : NCCLCommunicator
        ):
    recvbuff = tensor_to_c_ptr(dst)
    count = dst.numel().item()
    datatype = dtype2nccl(dst.dtype)
    C.ncclRecv(
        recvbuff,
        count,
        datatype,
        peer,
        comm.ptr,
        paddle.device.cuda.current_stream().cuda_stream
    )
    
def broadcast(
        src : paddle.Tensor,
        dst : paddle.Tensor,
        root : int,
        comm : NCCLCommunicator
    ):
    """NCCL API: `ncclBroadcast <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#ncclbroadcast>`_

    Args:
        src (torch.storage._StorageBase): Source buffer.
        dst (torch.storage._StorageBase): Destination buffer.
        root (int): Rank of the root.
        comm (NCCLCommunicator): NCCL communicator.
    
    The src and dst buffers must be the same size, type and on the same device.

    If src == dst, the operation is performed in-place.

    """

    assert src.dtype == dst.dtype, "send and recv buffers must be the same type"
    assert src.place.is_gpu_place() and dst.place.is_gpu_place(), "Tensors must be on GPU"

    sendbuff = tensor_to_c_ptr(src)
    recvbuff = tensor_to_c_ptr(dst)
    count = src.numel().item()
    datatype = dtype2nccl(src.dtype)

    assert dst.numel().item() == src.numel().item(), "Buffer size not aligned"
    C.ncclBroadcast(
        sendbuff, 
        recvbuff, 
        count, 
        datatype, 
        root, 
        comm.ptr, 
        paddle.device.cuda.current_stream().cuda_stream
    )

def reduce(
        src : paddle.Tensor,
        dst : paddle.Tensor,
        op : Literal["sum", "prod", "max", "min", "avg"],
        root : int,
        comm : NCCLCommunicator
    ):
    """NCCL API: `ncclReduce <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#ncclreduce>`_

    Args:
        src (torch.storage._StorageBase): Source buffer.
        dst (torch.storage._StorageBase): Destination buffer.
        op (Literal["sum", "prod", "max", "min", "avg"]): Reduction operation.
        root (int): Rank of the root.
        comm (NCCLCommunicator): NCCL communicator.
    
    The src and dst buffers must be the same size, type and on the same device.

    If src == dst, the operation is performed in-place.

    """
    assert src.dtype == dst.dtype, "send and recv buffers must be the same type"
    assert src.place.is_gpu_place() and dst.place.is_gpu_place(), "Tensors must be on GPU"

    sendbuff = tensor_to_c_ptr(src)
    recvbuff = tensor_to_c_ptr(dst)
    count = src.numel().item()
    datatype = dtype2nccl(src.dtype)
    operator = op2nccl(op)

    assert dst.numel().item() == src.numel().item(), "Buffer size not aligned"
    C.ncclReduce(
        sendbuff,
        recvbuff,
        count,
        datatype,
        operator,
        root,
        comm.ptr,
        paddle.device.cuda.current_stream().cuda_stream
    )

def allGather(
        src : paddle.Tensor,
        dst : paddle.Tensor,
        comm : NCCLCommunicator
    ):
    """NCCL API: `ncclAllGather <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#ncclallgather>`_

    Args:
        src (torch.storage._StorageBase): Source buffer.
        dst (torch.storage._StorageBase): Destination buffer.
        comm (NCCLCommunicator): NCCL communicator.
    
    The size of the dst buffer must be equal to the size of src buffer * world_size.

    The dst buffer is only used on rank root.

    """
    assert src.dtype == dst.dtype, "send and recv buffers must be the same time"
    assert src.place.is_gpu_place() and dst.place.is_gpu_place(), "Tensors must be on GPU"

    sendcount = src.numel().item()
    datatype = dtype2nccl(src.dtype)
    sendbuff = tensor_to_c_ptr(src)
    recvbuff = tensor_to_c_ptr(dst)
    # print(f"------------allGather----------{ dst.size}, {dst.numel().item()}, {sendcount}, \
    #         {src.size}, {src.numel().item()}")
    assert dst.numel().item() % sendcount == 0, "Buffer size not aligned"
    C.ncclAllGather(
        sendbuff, 
        recvbuff, 
        sendcount, 
        datatype, 
        comm.ptr, 
        paddle.device.cuda.current_stream().cuda_stream
    )


def reduceScatter(
        src : paddle.Tensor,
        dst : paddle.Tensor,
        op : Literal["sum", "prod", "max", "min", "avg"],
        comm : NCCLCommunicator
    ):
    """NCCL API: `ncclReduceScatter <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#ncclreducescatter>`_

    Args:
        src (torch.storage._StorageBase): Source buffer.
        dst (torch.storage._StorageBase): Destination buffer.
        op (Literal["sum", "prod", "max", "min", "avg"]): Reduction operation.
        comm (NCCLCommunicator): NCCL communicator.
    
    The size of the dst buffer must be equal to the size of src buffer / world_size.

    The dst buffer on rank `i` will contail the i-th block of the reduced result.

    """
    assert src.dtype == dst.dtype, "send and recv buffers must be the same time"
    assert src.place.is_gpu_place() and dst.place.is_gpu_place(), "Tensors must be on GPU"

    recvcount = dst.numel().item()
    datatype = dtype2nccl(src.dtype)
    operator = op2nccl(op)

    sendbuff = tensor_to_c_ptr(src)
    recvbuff = tensor_to_c_ptr(dst)

    # print("src.numel().item() % recvcount == 0", src.numel().item(), recvcount)
    assert src.numel().item() % recvcount == 0, "Buffer size not aligned"
    C.ncclReduceScatter(
        sendbuff,
        recvbuff,
        recvcount,
        datatype,
        operator,
        comm.ptr,
        paddle.device.cuda.current_stream().cuda_stream
    )

def groupStart():
    """
    NCCL API: `ncclGroupStart <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/group.html#ncclgroupstart>`_
    """
    C.ncclGroupStart()

def groupEnd():
    """
    NCCL API: `ncclGroupEnd <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/group.html#ncclgroupend>`_
    """
    C.ncclGroupEnd()
