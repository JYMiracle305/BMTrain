from typing import Generator, Iterable, List, Tuple
import paddle
from .block_layer import Block
from .parameter import DistributedParameter
from .global_var import config


def init_distributed_parameter(params: Iterable[paddle.Tensor]):
    """Init param of params which is instance of DistributedParameter using param._init_method.

    Args:
        params (Iterable[paddle.Tensor]): parameter tensors.

    """
    for param in params:
        if not isinstance(param, DistributedParameter):
            continue
        if param._init_method is None:
            continue
        with paddle.no_grad():
            partition_size = param.numel()
            global_size = partition_size * config["tp_zero_size"] * config["tp_size"]
            tmp_storage = param.storage_type()(global_size)
            tmp_tensor = paddle.tensor([], dtype=param.dtype).cuda()
            tmp_tensor.set_(tmp_storage, 0, param._tp_original_shape)

            param._init_method(tmp_tensor)
            if param._tp_mode and param._tp_split_dim >= 0:
                tensor_list = tmp_tensor.chunk(
                    config["tp_size"], dim=param._tp_split_dim
                )
                sub_tensor = tensor_list[config["topology"].tp_id].contiguous()
                tmp_tensor = paddle.empty(
                    sub_tensor.shape, device=param.device, dtype=sub_tensor.dtype
                )
                tmp_tensor.copy_(sub_tensor)

            if param._tp_mode:
                begin = config["tp_zero_rank"]
            else:
                begin = config["zero_rank"]
            end = begin + 1

            # Pytorch 1.11 changed the API of storage.__getitem__
            paddle.tensor([], dtype=param.dtype, device=param.device).set_(
                param.storage()
            )[:] = paddle.tensor([], dtype=param.dtype, device=param.device).set_(
                tmp_tensor.storage()
            )[
                partition_size * begin : partition_size * end
            ]
            # param.storage().copy_(tmp_storage[partition_size * config['rank'] : partition_size * (config['rank'] + 1)])


def iterate_parameters(model: paddle.nn.Layer):
    """
    Itterate over the parameters of the model.
    """
    for kw, val in model._parameters.items():
        if hasattr(val, "_in_block") and val._in_block:
            return []
        yield val


def init_parameters(model: paddle.nn.Layer):
    """
    Initialize the parameters of the model by calling the init_method of the distributed parameters.
    """

    modules = model.named_sublayers(include_self=True)
    for module_prefix, module in modules:
        if isinstance(module, Block):
            module.init_parameters()
        else:
            init_distributed_parameter(iterate_parameters(module))

    current_stream = paddle.device.cuda.current_stream()
    config["load_stream"].wait_stream(current_stream)


def grouped_parameters(
    model: paddle.nn.Layer,
) -> Generator[Tuple[str, List[paddle.Tensor]], None, None]:
    """
    Iterate over the parameters of the model grouped by the group name.
    This is similar to `paddle.nn.Layer.named_parameters()` .
    """

    ret: List[paddle.Tensor] = {}
    for module in model.modules():
        if isinstance(module, Block):
            for kw, params in module.grouped_parameters():
                if kw not in ret:
                    ret[kw] = []
                ret[kw].extend(params)
        else:
            for param in module._parameters.values():
                group = None
                if isinstance(param, DistributedParameter):
                    group = param.group
                if group not in ret:
                    ret[group] = []
                ret[group].append(param)
    for kw, val in ret.items():
        yield kw, val
