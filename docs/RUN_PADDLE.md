# Running BMTrain with PaddlePaddle

This guide describes how to install and run the current `bmtrain_paddle` branch in a GPU container. Tensor-parallel training has been verified on two NVIDIA A100 80GB GPUs with Python 3.12 and PaddlePaddle GPU 3.3.1.

## Requirements

- Linux container
- Python 3.12
- CUDA Toolkit, Ninja, and CMake
- NVIDIA GPU driver
- PaddlePaddle GPU 3.3.1

The verified environment uses a CUDA 13 container with A100 GPUs. Paddle's official cu126 wheel includes the CUDA 12.6 runtime and works with an NVIDIA driver that supports CUDA 12.x.

Install the GPU build:

```bash
python -m pip install paddlepaddle-gpu==3.3.1 --index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

Check single-GPU Paddle execution:

```bash
python -c "import paddle; paddle.utils.run_check()"
```

## Build and install

From the repository root, run:

```bash
BUILD_DOCKER_ENV=1 python -m pip install -e . --no-build-isolation --no-deps
```

CMake generates sm_80 code by default for A100 GPUs. For other architectures, set CMAKE_CUDA_ARCHITECTURES explicitly:

```bash
CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=80;86" BUILD_DOCKER_ENV=1 python -m pip install -e . --no-build-isolation --no-deps
```

Verify the import:

```bash
python -c "import bmtrain_paddle; print(bmtrain_paddle.__file__)"
```

## Run two-GPU tensor-parallel training

Enter the examples directory and start training:

```bash
cd example
python -m paddle.distributed.launch --log_dir=log-clean2 --gpus=0,1 train_by_paddle_multi.py
```

The example calls bmt.init_distributed(seed=0, tp_size=2). It performs model partitioning, forward computation, loss calculation, backward computation, and optimizer updates. It runs 1000 steps by default.
A successful run ends with output similar to the following on rank 0:

```text
Loss: 9.192007064819336
Weight delta: 1.7265625
word_emb.weight update delta: 1.7265625
| Iter:    999 | loss: 9.1920 average_loss: 9.1920 | lr: 3.1623e-05 scale: 1048576.0000 | time: 0.0563
```

Inspect the training log:

```bash
tail -80 example/log-clean2/workerlog.0
```

## Troubleshooting

Before starting a new job, confirm that processes from an earlier failed run have released their GPUs:

```bash
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
```

Terminate only the PIDs belonging to the failed job. Running jobs must not share the same NCCL communicators and log files.

If Paddle's multi-GPU check reports that child processes cannot find an entry point, the check was executed through standard input. Running a normal Python training file is unaffected.

## Supported paths

- Verified: Paddle-only package import and a complete two-GPU tensor-parallel training loop.
- Not verified: pure DDP/ZeRO, pipeline parallelism, and combined tensor/pipeline parallelism.
- Some tensor-parallel communication is synchronous for Paddle 3 compatibility. Re-enabling asynchronous overlap is a separate optimization task.
