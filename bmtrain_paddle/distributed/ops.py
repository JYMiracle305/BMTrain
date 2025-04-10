import paddle
from ..global_var import config
from ..nccl import allGather as ncclAllGather, recv
from ..nccl import allReduce as ncclAllReduce
from ..nccl import broadcast as ncclBroadcast
from ..nccl import reduceScatter as ncclReduceScatter
from ..nccl import send as ncclSend
from ..nccl import recv as ncclRecv
from ..nccl import commCount,commRank,NCCLCommunicator
DTYPE_LIST = [
    paddle.float64,
    paddle.float32,
    paddle.float16,
    paddle.int64,
    paddle.int32,
    paddle.int16,
    paddle.int8,
    paddle.bfloat16,
    paddle.bool
]
def send_activations(hidden_state, next_rank, comm):
    send_meta(hidden_state, next_rank, comm)
    ncclSend(hidden_state.storage(), next_rank, comm)

def recv_activations(prev_rank, comm):
    dtype, shape = recv_meta(prev_rank, comm)
    hidden_state = paddle.empty(shape, dtype=dtype, device="cuda")
    ncclRecv(hidden_state.storage(), prev_rank, comm)
    return hidden_state

def send_meta(x, next_rank, comm):
    meta_data = paddle.tensor(data=[0]*50, device="cuda", dtype='int')
    meta_data[0] = len(x.size())
    meta_data[1] = DTYPE_LIST.index(x.dtype)
    meta_data[2:len(x.size())+2] = paddle.tensor(x.size(), device="cuda", dtype='int')
    meta_data = meta_data.contiguous()
    ncclSend(meta_data.storage(), next_rank, comm)

def recv_meta(prev_rank, comm):
    meta_data = paddle.tensor(data=[0]*50, device="cuda", dtype='int')
    ncclRecv(meta_data.storage(), prev_rank, comm)
    n_dims = meta_data[0].item()
    dtype = DTYPE_LIST[meta_data[1].item()]
    shape = meta_data[2:n_dims+2].tolist()
    return dtype,shape

class OpBroadcast(paddle.autograd.PyLayer):

    @staticmethod
    def forward(src, root, comm = None):
        if comm is None:
            comm = config["comm"]
        outputs = paddle.empty_like(src, dtype = src.dtype, device = src.device)
        if src.place.is_gpu_place():
            outputs = outputs._to(place=src.place)
        ncclBroadcast(src, outputs, root, comm)
        return outputs, comm

    @staticmethod
    def backward(grad_output, *ctx_args):
        comm, root = ctx_args
        res = all_reduce(grad_output, "sum", comm)
        return res, None, None

def broadcast(src, root, comm=None):
    if not config["initialized"]:
        raise RuntimeError("BMTrain is not initialized")
    return OpBroadcast.apply(src, root, comm)

class OpAllGather(paddle.autograd.PyLayer):

    @staticmethod
    def forward(input : paddle.Tensor, comm = None):
        if comm is None:
            comm = config["comm"]
        world_size = commCount(comm)
        if not input.is_contiguous():
            input = input.contiguous()
        if input._offset() != 0 or input.numel() != input.size:
            input = input.clone()
        output = paddle.empty( (world_size,) + input.size(), dtype=input.dtype)
        if input.place.is_gpu_place():
            output = output._to(place=input.place)
        ncclAllGather(
            input._ptr(),
            output._ptr(),
            comm
        )
        return output, comm

    @staticmethod
    def backward(grad_output, comm):
        return grad_output[commRank(comm)], None

def all_gather(x : paddle.Tensor, comm = None):
    """Gathers the input tensor from all processes.

    Args:
        x (torch.Tensor): The input tensor of shape (...).
    
    Returns:
        torch.Tensor: The gathered tensor of shape (world_size, ...).
    """
    if not config["initialized"]:
        raise RuntimeError("BMTrain is not initialized")
    
    assert isinstance(x.place, paddle.CUDAPlace)
    return OpAllGather.apply(x, comm)

class OpReduceScatter(paddle.autograd.PyLayer):

    @staticmethod
    def forward(ctx, input : paddle.Tensor, op : str, comm : NCCLCommunicator = None):
        if comm is None:
            comm = config["comm"]
        rank = commRank(comm)
        assert input.shape[0] % commCount(comm) == 0, "The dimension 0 must be divisible by the number of communication processes"
        if not input.is_contiguous():
            input = input.contiguous()
        if input._offset() != 0 or input.numel() != input.size:
            input = input.clone()
        output_shape = (input.shape[0] // commCount(comm), *input.shape[1:])
        output = paddle.empty(output_shape, dtype=input.dtype)
        if input.place.is_gpu_place():
            output = output._to(place=input.place)
        ncclReduceScatter(
            input._ptr(),
            output._ptr(),
            op,
            comm
        )
        ctx = {
            'op': op,
            'comm': comm,
            'input_shape': input.shape,
            'world_size': commCount(comm)
        }
        if op in ["sum", "avg"]:
            pass
        elif op in ["max", "min"]:
            ctx['mask'] = (output == input[rank * output.shape[0]:(rank + 1) *output.shape[0]])
            # ctx.save_for_backward( output != input[rank * input.shape[0]:(rank + 1) * input.shape[0]] )
        elif op == "prod":
            ctx['prod_mask'] = (output / input[rank * output.shape[0]:(rank + 1) * output.shape[0]])
            # ctx.save_for_backward( output / input[rank * input.shape[0]:(rank + 1) * input.shape[0]] )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        op = ctx['op']
        comm = ctx['comm']
        world_size = ctx['world_size']
        with paddle.no_grad():
            grad_output = OpAllGather.apply(grad_output, comm).flatten(0,1)
        if op in ["max", "min", "prod"]:
            raise NotImplementedError("max min operation now do not support backward")
        else:
            if ctx.op == "avg":
                grad_output /= commCount(ctx.comm)
        # 形状重构（适配可能存在的非连续内存）
        grad_input = grad_input.reshape(ctx['input_shape']).contiguous()    

        return grad_output, None, None
       

def reduce_scatter(x : paddle.Tensor, op : str = "sum", comm = None):
    """Reduces the input tensor from all processes.

    Args:
        x (torch.Tensor): The input tensor of shape (world_size, ...).
        op (str): The reduction operation, one of "sum", "avg", "max", "min", "prod". Default: "sum".

    Returns:
        torch.Tensor: The reduced tensor of shape (...).
    
    """
    if not config["initialized"]:
        raise RuntimeError("BMTrain is not initialized")

    assert isinstance(x.place, paddle.CUDAPlace)
    return OpReduceScatter.apply(x, op, comm)

class OpAllReduce(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, input : paddle.Tensor, op : str, comm : NCCLCommunicator = None):
        if comm is None:
            comm = config["comm"]
        ctx.comm = comm
        if not input.is_contiguous():
            input = input.contiguous()
        if input._offset() != 0 or input.numel() != input.size:
            input = input.clone()
        output = paddle.empty( input.numel(), dtype=input.dtype)
        if input.place.is_gpu_place():
            output = output.cuda()

        print("!!!!!!!!!!!!!!!!!nccl all reduce--------------")

        # ncclAllReduce(
        #     input,
        #     output,
        #     op,
        #     comm
        # )
        ctx.op = op
        
        if op in ["sum", "avg"]:
            pass
        elif op in ["max", "min"]:
            ctx.save_for_backward( input != output )
        else:
            ctx.save_for_backward( output / input )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.op == "sum":
            return grad_output, None, None
        elif ctx.op == "avg":
            return grad_output / commCount(ctx.comm), None, None
        elif ctx.op in ["max", "min"]:
            mask = ctx['mask']
            grad_output = grad_output * mask.astype(grad_output.dtype)
            return grad_output, None, None
        else:
            return grad_output * ctx.saved_tensors[0], None, None

def all_reduce(x : paddle.Tensor, op : str = "sum", comm = None):
    """Reduces the input tensor from all processes.

    Args:
        x (torch.Tensor): The input tensor of shape (...).
        op (str): The reduction operation, one of "sum", "avg", "max", "min", "prod". Default: "sum".

    Returns:
        torch.Tensor: The reduced tensor of shape (...).
    
    """
    if not config["initialized"]:
        raise RuntimeError("BMTrain is not initialized")
    str(x.place) == "Place(gpu:0)"
    return OpAllReduce.apply(x, op, comm)


            
