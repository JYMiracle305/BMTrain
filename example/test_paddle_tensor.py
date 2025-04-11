import paddle
import torch

# tensor1 = paddle.randn([3, 4])
# len1 = tensor1.numel()
# print(tensor1.shape, len1)

# tensor2 = torch.randn(3, 4)
# size2 = tensor2.size()
# print(size2, tensor2.numel())

W = paddle.create_parameter(shape=[784, 200], dtype='float32')

print(W.type)