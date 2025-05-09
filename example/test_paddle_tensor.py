import paddle
import torch

# import bmtrain_paddle as bmt
# tensor1 = paddle.randn([3, 4])
# len1 = tensor1.numel()
# print(tensor1.shape, len1)

# tensor2 = torch.randn(3, 4)
# size2 = tensor2.size()
# print(size2, tensor2.numel())

# W = paddle.create_parameter(shape=[784, 200], dtype='float32')

# print(W.type)
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
    _in_block: bool

    def __init__(self, *args, **kwargs):
        # 空实现，阻止父类 Parameter.__init__ 被自动调用
        pass

    def __new__(
        cls,
        data: paddle.Tensor,
        requires_grad: bool = True,
        tp_mode: bool = False,
        tp_split_dim: int = -1,
    ):

        # === 关键步骤1：参数分片计算 ===
        num_of_elements = data.numel()
        
        # cuda_tensor = paddle.tensor([], dtype=data.dtype).cuda()
        # 获取通信组信息
        if tp_mode:
            comm = 1
        else:
            comm = 1
        world_size = 1
        rank = 0
        cuda_storage_size = 50

        original_shape = data.shape
        tp_original_shape = original_shape
        if tp_mode and tp_split_dim >= 0:
            tp_original_shape = list(original_shape)
            tp_original_shape[tp_split_dim] *= 1

        start_of_partition = cuda_storage_size * rank
        end_of_partition = min(num_of_elements, cuda_storage_size * (rank + 1))
        
        # FX: cuda_tensor_size < 0 if num_of_elements is too small
        cuda_tensor_size = max(end_of_partition - start_of_partition, 0)

        data_flat = paddle.flatten(data)

        local_shard = paddle.empty(tp_original_shape, dtype=data.dtype).cuda()
        if end_of_partition > start_of_partition:
            paddle.assign(
                data_flat[start_of_partition:end_of_partition], 
                local_shard[:end_of_partition-start_of_partition]
            )

        # ------------------- 参数构造 -------------------
        name = paddle.utils.unique_name.generate('dist_param')
        base_param = paddle.create_parameter(
            shape=local_shard.shape,
            dtype=local_shard.dtype,
            default_initializer=paddle.nn.initializer.Assign(local_shard),
            is_bias=False,
            attr=paddle.ParamAttr(name=name)
        )
        base_param.persistable = True
        base_param.stop_gradient = not requires_grad
        param = super().__new__(cls)
        param.__dict__.update(base_param.__dict__)
        # super(DistributedParameter, param).__init__(
        #     shape=local_shard.shape,
        #     dtype=local_shard.dtype,
        #     block=paddle.static.default_main_program().current_block()
        # )

        param._original_shape = original_shape
        param._start_partition = start_of_partition
        param._end_partition = end_of_partition
        # param._init_method = init_method
        param._in_block = False
        param._tp_original_shape = tp_original_shape
        param._tp_split_dim = tp_split_dim
        param._zero_comm = comm
        param._tp_mode = tp_mode
        # param._group = (group if not tp_mode else "tp")

        # ------------------- 初始化方法 -------------------
        # if init_method:
        #     init_method(param)

        return param

    def group(self):
        print("this is group function")

param = DistributedParameter(paddle.randn([128, 256]))
print(isinstance(param, DistributedParameter))  # 应输出 True

linear = paddle.nn.Linear(256, 512)
linear.weight = DistributedParameter(paddle.randn([256, 512]))
# print(linear.parameters())
# print("parameters  cat", linear._parameters['weight'].type, linear._parameters['weight'].stop_gradient, linear._parameters['weight']._original_shape)
print(linear.weight.group())

x = paddle.randn([32, 256]).cuda()
y = paddle.randn([32, 512]).cuda()
# 前向计算
output = linear(x)

# 计算损失
loss = paddle.nn.functional.mse_loss(output, y)
loss.backward()
print(linear.weight.grad)  # 应非空