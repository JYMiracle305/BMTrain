import paddle
from typing import Callable, Iterable, Optional
import numpy

class DistributedParameter(paddle.Tensor):
    _original_shape: paddle.Tensor.shape
    _start_partition: int
    _end_partition: int
    _init_method: Optional[Callable[["DistributedParameter"], None]]
    _in_block: bool
    _group: Optional[str]
    
    def __new__(
        self,
        data: paddle.Tensor,
        requires_grad: bool = True,
        init_method: Optional[Callable[["DistributedParameter"], None]] = None,
        group: Optional[str] = None,
        tp_mode: bool = False,
        tp_split_dim: int = -1,
    ):
        return paddle.create_parameter(shape=[5, 6], dtype=paddle.float32,
                    default_initializer=paddle.nn.initializer.XavierNormal())


# 创建一个全连接层，输出维度为 10，输入维度为 28，激活函数为 ReLU
layer = paddle.nn.Linear(28, 10)
# 创建一个形状为 [28, 10] 的张量，作为全连接层的权重
custom_weight = paddle.arange(280).reshape([28, 10]).astype('float32')

# 将自定义张量拷贝给全连接层的权重参数
weight_test = DistributedParameter(
    paddle.empty(shape=[5, 6], dtype=paddle.float32),
    tp_split_dim=0,
    tp_mode=True,
)

# 打印出全连接层权重的初始值和拷贝后的值
print("类型2", weight_test.type)
print("全连接层权重的初始值:")
print(layer.weight)

# 例如输出: Parameter containing:
# Tensor(shape=[28, 10], dtype='float32', place=Place(cpu), stop_gradient=False,
#        [[0.00000000e+00, 1.00000000e+00, 2.00000000e+00, ..., 2.70000000e+01],
#         [2.80000000e+01, 2.90000000e+01, 3.00000000e+00, ..., 5.40000000e+01],
#         ...,
#         [2.52000000e+02, 2.53000000e+02, 2.54000000e+02, ..., 2.79000000e+02]])

layer.weight = weight_test

print("类型1", layer.weight.type)
# 拷贝自定义张量给全连接层的权重参数后
print("\n拷贝自定义张量后的全连接层权重值:")
print(layer.weight)

# 例如输出: containing Parameter:
# Tensor(shape=[28, 10], dtype='float32', place=Place(cpu), stop_gradient=False,
#        [[0.00000000e+00, 1.00000000e+00, 2.00000000e+00, ..., 2.70000000e+01],
#         [2.80000000e+01, 2.90000000e+01, 3.00000000e+00, ..., 5.40000000e+01],
#         ...,
#         [2.52000000e+02, 2.53000000e+02, 2.54000000e+02, ..., 2.79000000e+02]])