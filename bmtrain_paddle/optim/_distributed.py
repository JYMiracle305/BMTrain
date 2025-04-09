import paddle
from ..distributed import all_reduce, all_gather


def state_dict_gather(state_dict):
    param_key = [
        p for param_group in state_dict["param_groups"] for p in param_group["params"]
    ]
    for k, v in state_dict["state"].items():
        if "step" in v:
            step = v["step"]

    for k in param_key:
        if k not in state_dict["state"]:
            state_dict["state"][k] = {
                "exp_avg": paddle.to_tensor([], dtype=paddle.float32, place=paddle.CUDAPlace(0)),
                "exp_avg_sq": paddle.to_tensor([], dtype=paddle.float32, place=paddle.CUDAPlace(0)),
                "_param_fp32": paddle.to_tensor([], dtype=paddle.float32, place=paddle.CUDAPlace(0)),
                "step": step,
            }
        v = state_dict["state"][k]
        for name, dtype in [
            ("exp_avg", paddle.float32),
            ("exp_avg_sq", paddle.float32),
            ("_param_fp32", paddle.float32),
        ]:
            if name in v:
                with paddle.no_grad():
                    numel = paddle.to_tensor(
                        v[name].numel(), dtype='int64', place=paddle.CUDAPlace(0)
                    )
                    max_numel = all_reduce(numel, op="max")
                    v_p = paddle.nn.functional.pad(
                        v[name], (0, max_numel - numel), value=-1e15
                    )
                    if max_numel > 0:
                        whole_state = all_gather(v_p.cuda()).flatten()
                        whole_state = whole_state[whole_state != -1e15]
                    v[name] = whole_state.contiguous().cpu()
    return state_dict
