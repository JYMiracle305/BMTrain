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
        dtype='float32'    #float16
    )

    bmt.init_parameters(model)

    bmt.print_rank("Model memory")
    # bmt.print_rank(paddle.cuda.memory_summary())
    bmt.synchronize()

    # data
    # generate dummy data for each rank
    paddle.manual_seed(1234)

    batch_size = 2
    seq_len = 512
    world_size = bmt.config["world_size"] if bmt.config["tp_size"] == 1 else bmt.config["tp_zero_size"]
    r = bmt.config["rank"] if bmt.config["tp_size"] == 1 else bmt.config["tp_zero_rank"] 

    for i in range(world_size):
        sent = np.random.randint(0, 10240, (batch_size, seq_len + 1))
        enc_length = np.random.randint(128, seq_len, (batch_size,)).long().cuda()
        enc_input = sent[:, :-1].long().cuda()
        targets = sent[:, 1:].long().cuda()
        mask = paddle.arange(seq_len).long().cuda()[None, :] < enc_length[:, None]
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

    optimizer = optim.AdamOffloadOptimizer(model.parameters(), weight_decay=1e-2)
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
            pos = paddle.arange(enc_input.size(1)).long().cuda().repeat(enc_input.size(0), 1)
            logits = model(
                enc_input,
                pos,
                pos < enc_length[:, None]
            )
            batch, seq_len, vocab_out_size = logits.size()

            if config['tp_size'] > 1:
                loss = loss_func(logits.view(batch * seq_len, vocab_out_size), targets.view(batch * seq_len))
            else:
                loss = loss_func(logits.float().view(batch * seq_len, vocab_out_size), targets.view(batch * seq_len))
        
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
                global_loss,
                avg_loss_recorder.value,
                lr_scheduler.current_lr,
                optim_manager.loss_scale,
                avg_time_recorder.value
            )
        )

        # save model
        if iteration % 1000 == 0:
            bmt.save(model, "ckpt-%d.pt" % iteration)
    
    bmt.save(model, "checkpoint.pt")

if __name__ == '__main__':
    main()
