import paddle
import paddle.nn.functional as F
import bmtrain as bmt


class OpLinear(paddle.autograd.PyLayer):
    @staticmethod
    def forward(x, weight, bias=None):
        ctx = {}
        ctx['saved_tensors'] = (x, weight, bias)
        return F.linear(x, weight, bias), ctx

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx['saved_tensors']
        grad_x = grad_weight = grad_bias = None
        if x.requires_grad:
            grad_x = grad_output.matmul(weight)
        if weight.requires_grad:
            dim = grad_output.dim()
            grad_weight = (
                grad_output.reshape(-1, grad_output.shape[-1])
                .t()
                .matmul(x.reshape(-1, x.shape[-1]))
            )
        if bias is not None and bias.requires_grad:
            grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(0)
        return grad_x, grad_weight, grad_bias


class Linear(bmt.DistributedModule):
    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, dtype=None
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = bmt.DistributedParameter(
            torch.empty(out_features, in_features, dtype=dtype, device="cuda"),
            init_method=paddle.nn.init.xavier_normal_,
        )
        if bias:
            self.bias = bmt.DistributedParameter(
                paddle.empty(out_features, dtype=dtype, device="cuda"),
                init_method=paddle.nn.init.zeros_,
            )
        else:
            self.register_parameter("bias", None)

    def forward(self, input):
        return OpLinear.apply(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )
