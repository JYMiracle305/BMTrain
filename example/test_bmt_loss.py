import paddle
import bmtrain_paddle as bmt

# 禁用分布式逻辑
# config["tp_size"] = 1
# config["tp_rank"] = 0

# 单卡前向传播
logits = paddle.randn([1024, 10240], dtype="float32")
target = paddle.randint(0, 10240, [1024], dtype="int64")
loss = bmt.loss.VPFusedCrossEntropy.apply(logits, target)
loss.backward()
print("Logits.grad:", logits.grad)  # 应非零