from typing import Tuple
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class Layernorm(nn.Layer):
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True, dtype=None):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)  # Convert to tuple if it's an integer
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = self.create_parameter(
                shape=self.normalized_shape,
                dtype=dtype if dtype else paddle.get_default_dtype(),
                default_initializer=nn.initializer.Constant(1.0)
            )
            self.bias = self.create_parameter(
                shape=self.normalized_shape,
                dtype=dtype if dtype else paddle.get_default_dtype(),
                default_initializer=nn.initializer.Constant(0.0)
            )
        else:
            self.weight = None
            self.bias = None

    def forward(self, input: paddle.Tensor) -> paddle.Tensor:
        return F.layer_norm(
            input, self.normalized_shape, weight=self.weight, bias=self.bias, epsilon=self.eps)

    def extra_repr(self) -> str:
        return '{normalized_shape}, eps={eps}, elementwise_affine={elementwise_affine}'.format(**self.__dict__)