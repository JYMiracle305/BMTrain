import torch

# 创建一个张量
tensor = torch.randn(3, 4, 5)  # 创建一个形状为 (3, 4, 5) 的随机张量

# 获取张量的形状
shape = tensor.shape
print("张量的形状是：", shape)

shape2 = tensor.size()
print("张量的形状是：", shape2)