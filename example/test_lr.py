import paddle
from paddle.optimizer.lr import NoamDecay

# 初始化调度器
# lr_scheduler = NoamDecay(
#     d_model=2560,
#     warmup_steps=8000,
#     learning_rate=4.0,
#     verbose=True
# )

lr_scheduler = paddle.optimizer.lr.NoamDecay(
    d_model=2560,
    warmup_steps=8000,
    learning_rate=1,
    last_epoch=-1,
    verbose=False
)
# 模拟训练循环
for step in range(10000):
    current_lr = lr_scheduler.get_lr()
    print(f"Step {step} 学习率: {current_lr}")
    lr_scheduler.step()