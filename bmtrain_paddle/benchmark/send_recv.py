from .. import nccl
from .shape import SHAPES
from ..global_var import config
from ..utils import print_rank
from .utils import format_size
import paddle
def send_recv():
    current_stream = paddle.device.cuda.current_stream()
    for shape in SHAPES:
        send_size = shape

        send_buffer = paddle.empty( send_size // 2, dtype='half', device="cuda" )
        recv_buffer = paddle.empty( send_size // 2, dtype='half', device="cuda" )
        
        start_evt = paddle.device.cuda.Event(enable_timing=True)
        end_evt = paddle.device.cuda.Event(enable_timing=True)

        current_stream.record_event(start_evt)
        nccl.groupStart()
        if config['rank'] in [0,2,4,6]:
            nccl.send(send_buffer.storage(), config['rank']+1, config['comm'])
        else:
            nccl.recv(recv_buffer.storage(), config['rank']-1, config['comm'])
        nccl.groupEnd()
        current_stream.record_event(end_evt)
        current_stream.synchronize()
        time_usage = start_evt.elapsed_time(end_evt)

        bw = shape / 1024 / 1024 / 1024 * 1000 / time_usage
        print_rank("Send Recv:\tsize {}\ttime: {:4.3f}\tbw: {:2.6f} GB/s".format(format_size(send_size), time_usage, bw))

