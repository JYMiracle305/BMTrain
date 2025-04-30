import paddle
import numpy as np
import os
import socket

def find_free_port():
    """动态获取空闲端口"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def run_allreduce_demo():
    """分布式训练函数（每个进程执行此函数）"""
    # 1. 初始化分布式环境
    # master_port = find_free_port()  # 动态获取主节点端口
    # rank = paddle.distributed.get_rank()
    # os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = str(master_port)
    # os.environ["PADDLE_TRAINER_ID"] = str(rank)
    # os.environ["PADDLE_TRAINERS_NUM"] = str(world_size)
    
    # 初始化并行环境
    paddle.distributed.init_parallel_env()

    rank = paddle.distributed.get_rank()
    world_size = paddle.distributed.get_world_size()

    # 2. 创建测试数据（每个 rank 的数据不同）
    data = paddle.to_tensor([rank + 1.0], dtype='float32')  # 示例数据：rank0=[1.0], rank1=[2.0]
    print(f"Rank {rank} | Before AllReduce: {data.numpy()}")

    # 3. 执行 AllReduce（求和操作）
    paddle.distributed.all_reduce(data, op=paddle.distributed.ReduceOp.SUM)

    # 4. 打印结果
    print(f"Rank {rank} | After AllReduce (sum): {data.numpy()}")

    # 5. 验证结果
    expected_sum = sum(range(1, world_size + 1))  # 1+2+...+world_size
    assert np.allclose(data.numpy(), [expected_sum]), "Result Mismatch!"

if __name__ == "__main__":
    # 启动 2 个进程（对应 2 个 GPU）
    # paddle.distributed.spawn(
    #     run_allreduce_demo,  # 主函数             # 参数：world_size=2
    #     nprocs=2             # 启动 2 个进程
    # )
    run_allreduce_demo()