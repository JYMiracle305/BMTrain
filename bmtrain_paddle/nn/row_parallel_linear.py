import paddle
# from paddle.nn.parameter import Parameter

import bmtrain_paddle as bmt
from bmtrain_paddle.global_var import config
from .parallel_linear_func import OpParallelLinear, ReduceType


class RowParallelLinear(bmt.DistributedModule):
    """Tensor Parallel use row partition for Linear.

    Args:
        in_features (int): in_features size.
        out_features (int): out_features size.
        bias (bool): whether use bias.
        dtype : data type.
        split_input (bool): whether split input before compute.
        all_reduce_output (bool): if true use all_reduce data after compute, or use reduce_scatter.
        async_chunks (int): chunk size for async.

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype=None,
        split_input=False,
        all_reduce_output=False,
        async_chunks=2,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.split_input = split_input
        self.all_reduce_output = all_reduce_output
        self.async_chunks = async_chunks
        tp_size = config["tp_size"]
        assert in_features % tp_size == 0
        self.in_features_per_partition = in_features // tp_size
        # self.weight = bmt.DistributedParameter(
        #     paddle.empty(
        #         self.out_features,
        #         self.in_features_per_partition,
        #         dtype=dtype,
        #     ).cuda(),
        #     init_method=paddle.nn.init.xavier_normal_,
        #     tp_split_dim=1,
        #     tp_mode=True,
        # )
        self.weight = paddle.create_parameter(shape=[self.in_features_per_partition, self.out_features], dtype=dtype,
            default_initializer=paddle.nn.initializer.XavierNormal())
        if bias:
            # self.bias = bmt.DistributedParameter(
            #     paddle.empty(self.out_features, dtype=dtype).cuda(),
            #     init_method=paddle.nn.init.zeros_,
            #     tp_split_dim=-1,
            #     tp_mode=True,
            # )
            self.bias = paddle.create_parameter(
                shape=[self.out_features], dtype=dtype,
                default_initializer=paddle.nn.initializer.Constant(0.0)
            )
        else:
            self.register_parameter("bias", None)

    def forward(self, input):
        print("---------------RowParallelLinear------------------")
        gather_input = self.split_input
        gather_output = False
        reduce_output_type = (
            ReduceType.ALL_REDUCE
            if self.all_reduce_output
            else ReduceType.REDUCE_SCATTER
        )
        out = OpParallelLinear.apply(
            input,
            self.weight,
            None,
            gather_input,
            gather_output,
            self.split_input,
            reduce_output_type,
            self.async_chunks,
        )
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features_per_partition, self.out_features, self.bias is not None
        )
