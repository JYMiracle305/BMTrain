import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.distributed import fleet

# 假设以下层已经用 PaddlePaddle 实现
from paddle.distributed.fleet.layers.mpu import ColumnParallelLinear, RowParallelLinear

class Feedforward(nn.Layer):
    def __init__(self, dim_model: int, dim_ff: int, bias: bool = True, dtype=None):
        super().__init__()

        tp_size = fleet.get_hybrid_communicate_group().get_model_parallel_world_size()
        tp_rank = fleet.get_hybrid_communicate_group().get_model_parallel_rank()

        if tp_size > 1:
            self.w_in = ColumnParallelLinear(dim_model, dim_ff, has_bias=bias)
            self.w_out = RowParallelLinear(dim_ff, dim_model, has_bias=bias)
        else:
            self.w_in = nn.Linear(dim_model, dim_ff, bias_attr=bias)
            self.w_out = nn.Linear(dim_ff, dim_model, bias_attr=bias)

        self.relu = nn.ReLU()

    def forward(self, input: paddle.Tensor) -> paddle.Tensor:
        return self.w_out(self.relu(self.w_in(input)))