import paddle
import torch

# 创建一维张量（长度为3）
x = paddle.to_tensor([[1, 2, 3],[4, 5, 6]], dtype=paddle.float32)
print(x.shape)  # 输出: [3]

x2 = paddle.max(x, axis=-1)
print(x2)

shape = list(x.shape)
print("shape = list(x.shape)", x.shape, shape)

logits = torch.tensor([[1, 2, 3],[4, 5, 6]], dtype=torch.float32)

shape = list(logits.shape)
print("shape = list(logitsx.shape)", x.shape, shape)

max_logits1 = torch.max(logits, dim=-1)
print(max_logits1)

max_logits = torch.max(logits, dim=-1)[0].float()
print(max_logits)

print("logits.shape[0]", logits.size(0), logits.shape[0])
logits_dim0_torch = torch.empty(logits.shape[0], dtype=torch.float32)
print("logits_dim0_torch", logits_dim0_torch)
logits_dim0 = paddle.empty([logits.shape[0]], dtype=paddle.float32)
print("logits_dim0", logits_dim0_torch)

print("max_logits", logits_dim0)


size_numerl = paddle.to_tensor([[1, 2, 3],[4, 5, 6]], dtype=paddle.float32)
print("size_numerl", size_numerl.size, size_numerl.numel().item())
