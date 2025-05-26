import paddle
import paddle.nn.functional as F
from paddle.distributed import fleet

from paddle.distributed.fleet.layers.mpu import ColumnParallelLinear, RowParallelLinear
import bmtrain_paddle as bmt
from bmtrain_paddle.nn import (
    Linear,
    ColumnParallelLinear,
    RowParallelLinear)
from bmtrain_paddle.global_var import config

class Feedforward(paddle.nn.Layer):
    def __init__(self, dim_model: int, dim_ff: int, bias: bool = True, dtype=None):
        super().__init__()

        if config['tp_size'] > 1:
            self.w_in = ColumnParallelLinear(dim_model, dim_ff, bias = bias, dtype=dtype)
            self.w_out = RowParallelLinear(dim_ff, dim_model, bias = bias, dtype=dtype)
        else:
            self.w_in = paddle.nn.Linear(dim_model, dim_ff, bias_attr=bias)
            self.w_out = paddle.nn.Linear(dim_ff, dim_model, bias_attr=bias)

        self.relu = paddle.nn.ReLU()

    def forward(self, input: paddle.Tensor) -> paddle.Tensor:
        return self.w_out(self.relu(self.w_in(input)))