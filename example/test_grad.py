import paddle

class CustomLinear(paddle.nn.Layer):
    def __init__(self, in_features, out_features):
        super().__init__()
        # 正确创建参数
        self.weight = self.create_parameter(
            shape=[in_features, out_features],
            dtype='float32',
            default_initializer=paddle.nn.initializer.XavierUniform()
        )
    
    def forward(self, x):
        # 使用 transpose 而不是 .T
        # weight_t = paddle.transpose(self.weight, perm=[1, 0])
        return paddle.nn.functional.linear(x, self.weight)
    
layer = CustomLinear(2, 3)
x = paddle.rand(shape=[4, 2], dtype=paddle.float32)
out = layer(x)
print(x, out)

x2 = paddle.randint(high=2, shape=[4, 2])
out2 = paddle.nn.functional.embedding(x2, layer.weight)
print(x2, out2)