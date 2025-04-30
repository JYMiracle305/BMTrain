import paddle
import os
import socket
from paddle.distributed import init_parallel_env

def find_free_port():
    with socket.socket() as s:
        s.bind(('', 0))  # 绑定到随机空闲端口
        return s.getsockname()[1]

def test_init():
    # 动态获取空闲端口
    master_port = find_free_port()
    # os.environ["PADDLE_MASTER"] = f"localhost:12345"
    # os.environ["PADDLE_TRAINER_ENDPOINTS"] = f"localhost:12345,localhost:12346"
    
    # 初始化分布式环境
    init_parallel_env()
    
    # 打印进程信息
    rank = paddle.distributed.get_rank()
    current_endpoint = os.getenv("PADDLE_CURRENT_ENDPOINT")  # 实际通信端口
    print(f"✅ Rank {rank} | Port:{current_endpoint.split(':')[-1]} | Initialized")

if __name__ == "__main__":
    # paddle.distributed.spawn(test_init, nprocs=2)
    test_init()