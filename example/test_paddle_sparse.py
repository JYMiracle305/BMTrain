import paddle
import paddle.nn as nn

# 最小测试模型
class TestModel(nn.Layer):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(10, 5, sparse=False)
    
    def forward(self, x):
        return self.emb(x).sum()

# 测试流程
model = TestModel()
opt = paddle.optimizer.Adam(parameters=model.parameters())
x = paddle.randint(0, 10, [4])  # 随机输入
loss = model(x)
loss.backward()
opt.step()  # 观察是否报错
