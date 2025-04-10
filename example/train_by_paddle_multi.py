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
from bmtrain_paddle import optim
from bmtrain_paddle.global_var import config
from bmtrain_paddle import inspect

def main():
    bmt.init_distributed(
        seed=0
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
        dtype=paddle.float16    #float16
    )

    bmt.init_parameters(model)

    bmt.print_rank("Model memory")
    # bmt.print_rank(paddle.cuda.memory_summary())
    # bmt.synchronize()

    # data
    # generate dummy data for each rank
    paddle.seed(1234)

    batch_size = 2
    seq_len = 512
    world_size = bmt.config["world_size"] if bmt.config["tp_size"] == 1 else bmt.config["tp_zero_size"]
    r = bmt.config["rank"] if bmt.config["tp_size"] == 1 else bmt.config["tp_zero_rank"] 

    for i in range(world_size):
        sent = paddle.randint(low = 0, high = 10240, shape = (batch_size, seq_len + 1))
        enc_length = paddle.randint(low = 128, high = seq_len, shape = (batch_size,), dtype='int64').cuda()
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
    optimizer = optim.AdamOffloadOptimizer(model.parameters(), weight_decay=1e-2)
    lr_scheduler = bmt.lr_scheduler.Noam(optimizer, start_lr=1e-3, warmup_iter=40, end_iter=1000, num_iter=0)

    optim_manager = optim.OptimManager(loss_scale=2**20)
    optim_manager.add_optimizer(optimizer, lr_scheduler)

    # bmt.synchronize()
    
    avg_time_recorder = bmt.utils.AverageRecorder()
    avg_loss_recorder = bmt.utils.AverageRecorder()

    for iteration in range(1000):
        # load data
        st = time.time()

        with inspect.inspect_tensor() as inspector:
            pos = paddle.arange(enc_input.shape[1], dtype='int64').unsqueeze(0).cuda().tile([enc_input.shape[0], 1])
            logits = model(
                enc_input,
                pos,
                pos < enc_length[:, None]
            )
            print("Logits 形状:", logits.shape)
            batch, seq_len, vocab_out_size = logits.shape

            if config['tp_size'] > 1:
                loss = loss_func(logits.reshape([batch * seq_len, vocab_out_size]), targets.reshape([batch * seq_len]))
            else:
                loss = loss_func(logits.astype('float32').reshape([batch * seq_len, vocab_out_size]), targets.reshape([batch * seq_len]))
        
            global_loss = bmt.sum_loss(loss).item()
            optim_manager.zero_grad()

            optim_manager.backward(loss)
        
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

        optim_manager.step()

        # record time and loss
        iteration_time = time.time() - st

        avg_time_recorder.record(iteration_time)
        avg_loss_recorder.record(global_loss)

        # print time and loss
        bmt.print_rank(
            "| Iter: {:6d} | loss: {:.4f} average_loss: {:.4f} | lr: {:.4e} scale: {:10.4f} | time: {:.4f}".format(
                iteration,
                global_loss.item(),
                avg_loss_recorder.value.item(),
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
