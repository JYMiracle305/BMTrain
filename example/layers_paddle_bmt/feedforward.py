import paddle
import bmtrain_paddle as bmt
from bmtrain_paddle.nn import (
    Linear,
    ColumnParallelLinear,
    RowParallelLinear)
from bmtrain_paddle.global_var import config

class Feedforward(bmt.DistributedModule):
    def __init__(self, dim_model: int, dim_ff: int, bias: bool = True, dtype=None):
        super().__init__()

        if config['tp_size'] > 1:
            self.w_in = ColumnParallelLinear(dim_model, dim_ff, bias = bias, dtype=dtype)
            self.w_out = RowParallelLinear(dim_ff, dim_model, bias = bias, dtype=dtype)
        else:
            self.w_in = Linear(dim_model, dim_ff, bias=bias, dtype=dtype)
            self.w_out = Linear(dim_ff, dim_model, bias=bias, dtype=dtype)

        self.relu = paddle.nn.ReLU()

    def forward(self, input: paddle.Tensor) -> paddle.Tensor:
        return self.w_out(self.relu(self.w_in(input)))