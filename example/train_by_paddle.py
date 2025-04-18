import time
import os
import numpy as np
import paddle
import paddle.nn as nn
import paddle.optimizer as optim
import paddle.distributed as dist
from paddle.profiler import Profiler
from paddle.vision.transforms import Compose, Resize, ToTensor, Normalize
from paddle.io import Dataset, DataLoader
import paddle.distributed.fleet as fleet
from models import GPT_paddle as GPT
import bmtrain_paddle as bmt
from paddle_GPT2 import GPT2LMHeadModel
def main():
    bmt.init_distributed(
        seed=0
    )
    # 初始化分布式环境
    use_distributed = dist.get_world_size() > 1
    if use_distributed:
        dist.init_parallel_env()

    world_size = dist.get_world_size()
    # 配置混合并行策略
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,  # 数据并行度
        "mp_degree": min(2, world_size),  # 模型并行度
        "pp_degree": 1,  # 流水线并行度
        "sharding_degree": 1  # Sharding 并行度
    }

    # 初始化 Fleet
    fleet.init(is_collective=True, strategy=strategy)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"rank: {rank} world_size: {world_size}")
    # 模型定义
    # model = GPT2LMHeadModel(
    model = GPT(
        num_layers=8,
        vocab_size=10240,
        dim_model=2560,
        dim_head=80,
        num_heads=32,
        dim_ff=8192,
        max_distance=1024,
        bias=True,
        dtype='float32'
    )

    # 数据生成
    paddle.seed(1234)

    batch_size = 2
    seq_len = 512

    # 生成虚拟数据
    sent = np.random.randint(0, 10240, (batch_size, seq_len + 1))
    enc_length = np.random.randint(128, seq_len, (batch_size,))
    enc_input = paddle.to_tensor(sent[:, :-1], dtype='int64')
    targets = paddle.to_tensor(sent[:, 1:], dtype='int64')
    mask = paddle.arange(seq_len).unsqueeze(0) < paddle.to_tensor(enc_length).unsqueeze(1)
    targets = paddle.where(mask, targets, paddle.full_like(targets, -100, dtype='int64'))

    # 损失函数
    loss_func = nn.CrossEntropyLoss(ignore_index=-100)

    # 优化器
    lr_scheduler = optim.lr.NoamDecay(
        d_model=2560,
        warmup_steps=4000,
        learning_rate=4,
        last_epoch=-1,
        verbose=False
    )
    # paddle.set_default_dtype('float32')

    optimizer = optim.Adam(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=1e-2,
        multi_precision=True
    )
    # for step in range(10000):
    #     current_lr = lr_scheduler.get_lr()
    #     print(f"Step {step} 学习率: {current_lr}")
    #     lr_scheduler.step()
    scaler = paddle.amp.GradScaler(init_loss_scaling=1024.0)

    # 分布式训练
    if world_size > 1:
        model = paddle.DataParallel(model)

    # 训练循环
    avg_time_recorder = AverageRecorder()
    avg_loss_recorder = AverageRecorder()

    for iteration in range(1000):
        st = time.time()

        pos = paddle.arange(enc_input.shape[1]).unsqueeze(0)
        pos = paddle.tile(pos, [batch_size, 1])
        mask = pos < paddle.to_tensor(enc_length).unsqueeze(1)

        with paddle.amp.auto_cast():
            logits = model(enc_input, pos, mask)
            loss = loss_func(logits.reshape([-1, logits.shape[-1]]), targets.reshape([-1]))

        # print(f"词嵌入输出范围: [{model.word_emb(enc_input).min().item()}, {model.word_emb(enc_input).max().item()}]")

        # # print(f"当前梯度缩放因子: {scaler.get_loss_scale()}")
        # print(f"平均损失值: {loss.item()}")
        scaler.scale(loss).backward()


        # for name, param in list(model.named_parameters())[:10]:
        #     if param.grad is None:
        #         print(f"参数 {name} 梯度为 None")
        #     else:
        #         print(f"参数 {name} 梯度均值: {param.grad.mean().item()}")

        weight_before = model.word_emb.weight.clone()
        scaler.step(optimizer)

        weight_after = model.word_emb.weight.clone()
        print(f"词嵌入权重变化: {(weight_after - weight_before).abs().mean().item()}")
        
        scaler.update()

        global_loss = loss.numpy()
        # 记录时间与损失
        iteration_time = time.time() - st
        avg_time_recorder.record(iteration_time)
        avg_loss_recorder.record(global_loss)

        print(
            "| Iter: {:6d} | loss: {} average_loss: {:.4f} | lr: {} | time: {:.4f}".format(
                iteration,
                global_loss,
                avg_loss_recorder.value,
                lr_scheduler.get_lr(),
                avg_time_recorder.value
            )
        )
        lr_scheduler.step()
        optimizer.clear_grad()

    #     if iteration % 1000 == 0:
    #         paddle.save(model.state_dict(), f"ckpt-{iteration}.pdparams")

    # paddle.save(model.state_dict(), "checkpoint.pdparams")

class AverageRecorder:
    def __init__(self):
        self.sum = 0
        self.count = 0

    def record(self, value):
        self.sum += value
        self.count += 1

    @property
    def value(self):
        return self.sum / self.count if self.count != 0 else 0

if __name__ == '__main__':
    main()