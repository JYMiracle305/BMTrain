from typing import Callable, Iterable, Optional
import paddle
from paddle import device
from .utils import round_up
from .global_var import config
from . import nccl
from .distributed import all_gather


class DistributedParameter(paddle.Tensor):
    r"""
    DistributedParameter is a subclass of paddle.Tensor.

    It scatters the tensor to all the nodes and gathers them when needed.

    Args:
        data (Tensor): parameter tensor.
        requires_grad (bool, optional): if the parameter requires gradient.
        init_method (Callable[['DistributedParameter'], None], optional): the method to initialize the parameter.
        group (str, optional): the group name of the parameter.

    **Note**: DistributedParameter must be on the CUDA device. It will transfer the data to device automatically when `__init__` called.

    """

    _original_shape: paddle.Tensor.shape
    _start_partition: int
    _end_partition: int
    _init_method: Optional[Callable[["DistributedParameter"], None]]
    _in_block: bool
    _group: Optional[str]

    def __new__(
        cls,
        data: paddle.Tensor,
        requires_grad: bool = True,
        init_method: Optional[Callable[["DistributedParameter"], None]] = None,
        group: Optional[str] = None,
        tp_mode: bool = False,
        tp_split_dim: int = -1,
    ):
        if not config["initialized"]:
            raise RuntimeError("BMTrain is not initialized")

        # === 关键步骤1：参数分片计算 ===
        num_of_elements = data.numel()
        
        # cuda_tensor = paddle.tensor([], dtype=data.dtype).cuda()
        # 获取通信组信息
        if tp_mode:
            comm = config["tp_zero_comm"]
        else:
            comm = config["zero_comm"]
        world_size = nccl.commCount(comm)
        rank = nccl.commRank(comm)

        original_shape = data.size()
        tp_original_shape = original_shape
        if tp_mode and tp_split_dim >= 0:
            tp_original_shape = list(original_shape)
            tp_original_shape[tp_split_dim] *= config["tp_size"]

        # 计算分片大小（对齐到 512 字节）
        element_size = data.element_size()
        align_size = 512 // element_size
        chunk_size = (num_of_elements + world_size - 1) // world_size
        chunk_size = (chunk_size + align_size - 1) // align_size * align_size

        # === 关键步骤2：显存预分配 ===
        # 直接在 GPU 上创建分片存储
        place = paddle.CUDAPlace(paddle.distributed.ParallelEnv().dev_id)
        storage = paddle.empty([chunk_size], dtype=data.dtype).cuda(rank)
        
        # === 关键步骤3：数据分片拷贝 ===
        start = chunk_size * rank
        end = min(start + chunk_size, num_elements)
        storage_slice = storage[0:end-start]
        
        if data.place.is_gpu_place():
            # 同设备直接拷贝
            storage_slice.copy_(data.flatten()[start:end], False)
        else:
            # 跨设备异步拷贝
            with paddle.device.CUDAPlace(place.get_device()):
                tmp = paddle.to_tensor(data.flatten()[start:end], place=place)
                storage_slice.copy_(tmp, False)

        # === 关键步骤4：构造子类实例 ===
        instance = super().__new__(cls, storage_slice)
        instance.stop_gradient = not requires_grad

        # === 步骤6：元数据记录 ===
        instance._original_shape = data.shape
        instance._tp_mode = tp_mode
        instance._split_dim = tp_split_dim
        # instance._group = mp_group if tp_mode else dp_group

        # === 步骤7：参数初始化 ===
        if init_method is not None:
            init_method(instance)
        else:
            # 默认 Xavier 初始化
            paddle.nn.initializer.XavierNormal()(instance)

        return instance

    @property
    def group(self):
        """The group name of the distributed parameter."""

        return self._group

    def gather(self) -> paddle.Tensor:
        """Gather the data from ZeRO distributed nodes.

        Return:
            paddle.Tensor: The gathered data.

        """
        with device.cuda.stream(config["load_stream"]):
            output_tensor = OpAllGather.apply(self)
        current_stream = device.cuda.current_stream()
        output_tensor.record_stream(current_stream)
        current_stream.wait_stream(config["load_stream"])
        return output_tensor

    def gather_all(self) -> paddle.tensor:
        """Gather the data from ZeRO and Tensor Parallel distributed nodes.

        Return:
            paddle.Tensor: The gathered data.

        """
        zero_param = self.gather()
        if config["tp_size"] > 1 and self._tp_split_dim >= 0:
            output_tensor = all_gather(zero_param, config["tp_comm"])
            if self._tp_split_dim == 1:
                output_list = output_tensor.chunk(config["tp_size"], dim=0)
                output = paddle.concat(output_list, dim=output_list[0].dim() - 1).flatten(
                    0, 1
                )
                return output
            else:
                return output_tensor.flatten(0, 1)
        else:
            return zero_param

    def tp_gather(self) -> paddle.tensor:
        """Gather the data from Tensor Parallel distributed nodes.

        Return:
            torch.Tensor: The gathered data.

        """
        if config["tp_size"] > 1 and self._tp_split_dim >= 0:
            output_tensor = all_gather(self, config["tp_comm"])
            if self._tp_split_dim == 1:
                output_list = output_tensor.chunk(config["tp_size"], dim=0)
                output = paddle.concat(output_list, dim=output_list[0].dim() - 1).flatten(
                    0, 1
                )
                return output
            else:
                return output_tensor.flatten(0, 1)
        else:
            return self

    def _copy_data(self, data: paddle.Tensor):
        """Copy data to self.data."""
        self.data.copy_(data.view(-1)[self._start_partition : self._end_partition])


class OpAllGather(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, value: DistributedParameter):
        assert isinstance(value, DistributedParameter)
        comm = value._zero_comm  # config['zero_comm']
        world_size = nccl.commCount(comm)
        ctx.comm = comm
        ctx.world_size = world_size

        partition_size = value.storage().size()
        global_size = partition_size * world_size

        storage = value.storage_type()(global_size)

        nccl.allGather(value.storage(), storage, comm)

        output_tensor = paddle.tensor([], dtype=value.dtype).cuda()
        output_tensor.set_(storage, 0, value._original_shape)

        ctx.partition_size = partition_size
        ctx.tensor_size = value.size(0)
        return output_tensor

    @staticmethod
    def backward(ctx, grad_output: paddle.Tensor):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()

        grad_storage = grad_output.storage_type()(ctx.partition_size)
        grad_output_storage = grad_output.storage()
        if grad_output_storage.size() == ctx.partition_size * ctx.world_size:
            pass
        else:
            grad_output_storage.resize_(ctx.partition_size * ctx.world_size)
        nccl.reduceScatter(grad_output_storage, grad_storage, "sum", ctx.comm)
        grad_tensor = paddle.tensor([], dtype=grad_output.dtype).cuda()
        grad_tensor.set_(grad_storage, 0, (ctx.tensor_size,))
        return grad_tensor


class ParameterInitializer:
    """
    ParameterInitializer is a helper class that is used to initialize the distributed parameters.

    Similar to functools.partial .

    """

    def __init__(self, func: Callable, *args, **kwargs) -> None:
        self.func = func
        self._args = args
        self._kwargs = kwargs

    def __call__(self, param: DistributedParameter):
        self.func(param, *self._args, **self._kwargs)
