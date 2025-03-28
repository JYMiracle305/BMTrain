from .. import nccl
from .shape import SHAPES
from ..global_var import config
from ..utils import round_up, print_rank
from .utils import format_size
import paddle

def all_gather():
    current_stream = paddle.device.cuda.current_stream()
    for shape in SHAPES:
        global_size = round_up(shape, config['world_size'] * 2)
        partition_size = global_size // config['world_size']

        partition_tensor = paddle.empty( partition_size // 2, dtype='half', device="cuda" )
        global_tensor = paddle.empty( global_size // 2, dtype='half', device="cuda" )
        
        start_evt = paddle.device.cuda.Event(enable_timing=True)
        end_evt = paddle.device.cuda.Event(enable_timing=True)

        current_stream.record_event(start_evt)
        nccl.allGather(partition_tensor.storage(), global_tensor.storage(), config['comm'])
        current_stream.record_event(end_evt)
        current_stream.synchronize()
        time_usage = start_evt.elapsed_time(end_evt)

        bw = global_size / 1024 / 1024 / 1024 * 1000 / time_usage
        print_rank("All gather:\tsize {}\ttime: {:4.3f}\tbw: {:2.6f} GB/s".format(format_size(global_size), time_usage, bw))

