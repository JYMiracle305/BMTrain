from typing import Tuple
import paddle
import paddle.nn.functional as F
import bmtrain_paddle as bmt

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
        if self.elementwise_affine:
            # self.weight = self.create_parameter(
            #     shape=self.normalized_shape,
            #     dtype=dtype if dtype else paddle.get_default_dtype(),
            #     default_initializer=paddle.nn.initializer.Constant(1.0)
            # )
            # self.bias = self.create_parameter(
            #     shape=self.normalized_shape,
            #     dtype=dtype if dtype else paddle.get_default_dtype(),
            #     default_initializer=paddle.nn.initializer.Constant(0.0)
            # )
            self.weight = bmt.DistributedParameter(paddle.empty(self.normalized_shape, dtype=dtype).cuda())
            self.bias = bmt.DistributedParameter(paddle.empty(self.normalized_shape, dtype=dtype).cuda())
        else:
            self.create_parameter('weight', None)
            self.create_parameter('bias', None)

    def forward(self, input: paddle.Tensor) -> paddle.Tensor:
        return F.layer_norm(
            input, self.normalized_shape, weight=self.weight, bias=self.bias, epsilon=self.eps)

    def extra_repr(self) -> str:
        return '{normalized_shape}, eps={eps}, elementwise_affine={elementwise_affine}'.format(**self.__dict__)