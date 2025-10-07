import paddle

class MyCustomLayer(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, input, weight,
                bias,
                gather_input,
                gather_output,
                split_input,
                reduce_output_type,
                async_gather_chunks,
        ):
        ctx.save_for_backward(input, weight)
        ctx.gather_output = gather_output
        ctx.split_input = split_input
        ctx.gather_input = gather_input
        ctx.reduce_output_type = reduce_output_type
        ctx.async_gather_chunks = async_gather_chunks

        output = paddle.matmul(input, weight)
        # if bias is not None:
        #     output += bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensor()
        print (ctx.gather_output,
            ctx.split_input,
            ctx.gather_input,
            ctx.reduce_output_type,
            ctx.async_gather_chunks)
        grad_input = grad_weight = grad_bias = None

        if not input.stop_gradient:
            grad_input = paddle.matmul(grad_output, weight, transpose_y=True)

        if not weight.stop_gradient:
            grad_weight = paddle.matmul(input, grad_output, transpose_x=True)

        # if bias is not None and not bias.stop_gradient:
        #     grad_bias = grad_output.sum(axis=0)

        # 返回 8 个梯度会报错
        # return (grad_input, grad_weight, grad_bias, None, None, None, None, None)
        # 返回 2 个梯度 OK
        return (grad_input, grad_weight)

input_data = paddle.randn([1, 16384, 11008], dtype='float16')  # 形状 [batch_size, in_features]
input_data.stop_gradient = False
weight = paddle.randn([11008, 4096], dtype='float16')      # 形状 [in_features, out_features]
weight.stop_gradient = False
# bias = paddle.zeros([4], dtype='float32')
# bias.stop_gradient = True

# 前向传播
output = MyCustomLayer.apply(input_data, weight, None, True, False, False, None, 10)

# 计算损失
loss = output.sum()
print("loss损失值", loss)
# 反向传播
loss.backward()

print("输入梯度：\n", input_data.grad)
print("权重梯度：\n", weight.grad)


# 手动计算预期梯度
expected_grad_input = paddle.matmul(paddle.ones_like(output), weight, transpose_y=True)
expected_grad_weight = paddle.matmul(input_data, paddle.ones_like(output), transpose_x=True)

# 检查梯度是否匹配
print("输入梯度是否匹配:", paddle.allclose(input_data.grad, expected_grad_input).item())  # 应为 True
print("权重梯度是否匹配:", paddle.allclose(weight.grad, expected_grad_weight).item())    # 应为 True