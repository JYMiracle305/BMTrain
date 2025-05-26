from typing import Tuple
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import bmtrain_paddle as bmt
from bmtrain_paddle.global_var import config

class Layernorm(bmt.DistributedModule):
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True,
                dtype=None) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        world_size = config["world_size"] 
        if self.elementwise_affine:
            self.weight = bmt.DistributedParameter(paddle.empty(self.normalized_shape, dtype=dtype).cuda())
            print("--------------------self.weight----------------", self.weight.shape)
            self.bias = bmt.DistributedParameter(paddle.empty(self.normalized_shape, dtype=dtype).cuda())
            print("--------------------self.bias----------------", self.bias.shape)
            print("------------Layernorm-------------", self.normalized_shape)
            # self.weight = self.create_parameter(shape=self.normalized_shape, dtype=dtype,
            #         default_initializer=paddle.nn.initializer.XavierNormal())
            # self.bias = self.create_parameter(shape=self.normalized_shape, dtype=dtype,
            #         default_initializer=paddle.nn.initializer.Constant(0.0))
        else:
            self.weight = None
            self.bias = None

    def forward(self, input: paddle.Tensor) -> paddle.Tensor:
        return F.layer_norm(
            input, self.normalized_shape, weight=self.weight, bias=self.bias, epsilon=self.eps)

    def extra_repr(self) -> str:
        return '{normalized_shape}, eps={eps}, elementwise_affine={elementwise_affine}'.format(**self.__dict__)