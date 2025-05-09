import torch

stream = torch.cuda.current_stream()
print("torch", stream, stream.cuda_stream)  # 输出可能是 0 或动态地址（如 140423041155072）

import paddle

stream = paddle.device.cuda.current_stream()
print("paddle", stream, stream.cuda_stream)  # 输出动态地址（如 94342331836288）