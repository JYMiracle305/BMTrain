import paddle
import paddle.nn.functional as F
import bmtrain_paddle as bmt


class OpLinear(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, x, weight, bias=None):
        ctx.save_for_backward(x, weight, bias)
        print(f"forward x {x.shape} weight {weight.shape} bias {bias.shape}")
        return F.linear(x, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        saved_tensors = ctx.saved_tensor()
        x, weight, bias = ctx.saved_tensor()
        grad_x = grad_weight = grad_bias = None
        # if x.requires_grad:
        print(f"backward grad_output {grad_output.shape} weight {weight.shape}")
        grad_x = grad_output.matmul(weight.T)
        # if weight.requires_grad:
        dim = grad_output.dim()

        grad_weight = (
            grad_output.reshape([-1, grad_output.shape[-1]])
            .t()
            .matmul(x.reshape([-1, x.shape[-1]]))
        )
        if bias is not None:
            #  and bias.requires_grad
            grad_bias = grad_output.reshape([-1, grad_output.shape[-1]]).sum(0)
        return grad_x, grad_weight, grad_bias


class Linear(bmt.DistributedModule):
    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, dtype=None
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        test_tensor = paddle.empty([in_features, out_features], dtype=dtype).cuda()
        # self.weight = bmt.DistributedParameter(
        #     paddle.empty([in_features, out_features], dtype=dtype).cuda(),
        #     init_method=paddle.nn.initializer.XavierNormal(),
        # )
        self.weight = paddle.create_parameter(shape=[in_features, out_features], dtype=dtype,
            default_initializer=paddle.nn.initializer.XavierNormal())
        if bias:
            self.bias = paddle.create_parameter(
                shape=[out_features], dtype=dtype,
                default_initializer=paddle.nn.initializer.Constant(0.0)
            )
        else:
            self.bias = paddle.create_parameter(shape=[0], dtype=dtype)
    def forward(self, input):
        return OpLinear.apply(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )