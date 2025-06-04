import time
import os
import numpy as np
import paddle
from models import GPT_paddle_bmt as GPT

import bmtrain_paddle as bmt
from bmtrain_paddle import optim
from bmtrain_paddle.global_var import config
from bmtrain_paddle import inspect

def main():
    bmt.init_distributed(
        seed=0,
        tp_size=2,
    )

    model = GPT(
        num_layers=8,
        vocab_size=10240, 
        dim_model=2560,
        dim_head=80,
        num_heads=32,
        dim_ff=8192,
        max_distance=1024,
        bias=True,
        dtype=paddle.float16,    #float16
    )

    bmt.init_parameters(model)
    # bmt.print_rank(
    #     inspect.format_summary(
    #         inspect.inspect_model(model, "*")
    #     )
    # )
    bmt.print_rank("Model memory")
    # bmt.print_rank(paddle.cuda.memory_summary())
    bmt.synchronize()

    # generate dummy data for each rank
    paddle.seed(1234)

    batch_size = 2
    seq_len = 512
    world_size = bmt.config["world_size"] if bmt.config["tp_size"] == 1 else bmt.config["tp_zero_size"]
    r = bmt.config["rank"] if bmt.config["tp_size"] == 1 else bmt.config["tp_zero_rank"] 

    for i in range(world_size):
        sent = paddle.randint(low = 0, high = 10240, shape = (batch_size, seq_len + 1))
        enc_length = paddle.randint(low = 400, high = seq_len, shape = (batch_size,), dtype='int64').cuda()
        enc_input = sent[:, :-1].astype('int64').cuda()
        targets = sent[:, 1:].astype('int64').cuda()
        mask = paddle.arange(seq_len).astype('int64').cuda().unsqueeze(0) < enc_length.unsqueeze(1)
        targets = paddle.where(
            mask,
            targets,
            paddle.full_like(targets, -100, dtype='int64')
        )

        if i == r:
            break
    
    if config['tp_size'] > 1:
        loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100, parallel=True)
    else:
        loss_func = paddle.nn.CrossEntropyLoss(ignore_index=-100)
    # print(f"model.parameters() {model.parameters()}")
    # bmt优化器
    optimizer = optim.AdamOptimizer(model.parameters(), weight_decay=1e-2)

    lr_scheduler = bmt.lr_scheduler.Noam(optimizer, start_lr=1e-3, warmup_iter=40, end_iter=1000, num_iter=0)

    optim_manager = optim.OptimManager(loss_scale=2**20)
    optim_manager.add_optimizer(optimizer, lr_scheduler)

    bmt.synchronize()

    avg_time_recorder = bmt.utils.AverageRecorder()
    avg_loss_recorder = bmt.utils.AverageRecorder()

    for iteration in range(1000):
        # load data
        st = time.time()

        with inspect.inspect_tensor() as inspector:
            pos = paddle.arange(enc_input.shape[1], dtype=paddle.int64).unsqueeze(0).cuda().tile([enc_input.shape[0], 1])
            # print("输入形状", enc_input, pos)
            # print("all-------------------------计算前梯度状态:", paddle.is_grad_enabled())
            logits = model(
                enc_input,
                pos,
                pos < enc_length[:, None]
            )
            print("Logits 形状:", logits.shape)
            batch, seq_len, vocab_out_size = logits.shape

            # print(f"Iter {iteration} 当前学习率: {optimizer.get_lr()}")

            print("--------标签最大值:------", paddle.max(targets).item())
            print("--------标签最小值:------", paddle.min(targets).item())
            print("logits 形状为", batch * seq_len, vocab_out_size)

            if config['tp_size'] > 1:
                loss = loss_func(logits.reshape([batch * seq_len, vocab_out_size]), targets.reshape([batch * seq_len]))
            else:
                loss = loss_func(logits.astype(paddle.float32).reshape([batch * seq_len, vocab_out_size]), targets.reshape([batch * seq_len]))

            global_loss = bmt.sum_loss(loss).item()

            optim_manager.zero_grad()
            optim_manager.backward(loss)    # loss.backward()

            print(f"Loss 值: {loss} {loss.item()} {global_loss}")
            # 添加梯度检查代码
            # for name, param in list(model.named_parameters())[:10]:
            #     if param.grad is None:
            #         print(f"参数 {name} 梯度为 None")
            #     else:
            #         print(f"参数 {name} 梯度均值: {param.grad.mean().item()}")
            # # 检查梯度是否存在
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(f"参数 {name} 梯度均值: {param.grad.mean().item()}")
            #     else:
            #         print(f"参数 {name} 梯度为 None")

            # 添加梯度检查代码
            # for name, param in model.named_parameters():
            #     if param.grad is None:
            #         print(f"参数 {name} 无梯度！")
            #     else:
            #         grad_norm = paddle.linalg.norm(param.grad).item()
            #         print(f"参数 {name} 梯度范数: {grad_norm:.6f} (应 > 0)")

        # print inspected tensors in the forward & backward pass
        # print parameters of the model
        if iteration % 100 == 0:
            bmt.print_rank(
                inspect.format_summary(
                    inspector.get_summary()
                )
            )
            bmt.print_rank(
                inspect.format_summary(
                    inspect.inspect_model(model, "*")
                )
            )

        param_samples = []
        for name, param in model.named_parameters():
            if "word_emb" in name:  # 选择关键层（如Embedding）
                param_samples.append((name, param.clone()))

        # 检查参数是否更新
        weight_before = model.word_emb.weight.clone()
        optim_manager.step()
        weight_after = model.word_emb.weight.clone()
        # print(f"weight_before {weight_before}, weight_after {weight_after}")
        print(f"权重变化: {(weight_after - weight_before).abs().sum().item()}")

        for name, param_before in param_samples:
            param_after = model.state_dict()[name]
            delta = (param_after - param_before).abs().sum().item()
            print(f"参数 {name} 更新量: {delta} (应 > 0)")

        # record time and loss
        iteration_time = time.time() - st

        avg_time_recorder.record(iteration_time)
        avg_loss_recorder.record(global_loss)

        # print time and loss
        bmt.print_rank(
            "| Iter: {:6d} | loss: {:.4f} average_loss: {:.4f} | lr: {:.4e} scale: {:10.4f} | time: {:.4f}".format(
                iteration,
                global_loss,
                avg_loss_recorder.value,
                lr_scheduler.current_lr,
                optim_manager.loss_scale,
                avg_time_recorder.value
            )
        )

        # save model
        # if iteration % 1000 == 0:
        #     bmt.save(model, "ckpt-%d.pt" % iteration)
    
    # bmt.save(model, "checkpoint.pt")

if __name__ == '__main__':
    main()
