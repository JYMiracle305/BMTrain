import paddle
from ..global_var import config
from . import _function as F
from .. import C
from .. import nccl
import inspect
from ..utils import check_torch_version
from copy import deepcopy
from itertools import chain
from collections import defaultdict


class AdamOptimizer(paddle.optimizer.Optimizer):
    """
    Adam optimizer support fp16 and bf16.
    """

    _bmtrain_optimizer = True

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        hold_steps=0,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.lr = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.weight_decay = weight_decay
        #params = list(params)

        # print("\n2222222222222222222222222222=== 原始参数信息 ===")
        # if len(params) == 0:
        #     raise ValueError("参数列表为空！请检查模型是否注册了参数")
        # for idx, param in enumerate(params):
        #     print(f"参数 {idx}:")
        #     print(f"  名称: {param.name}")
        #     print(f"  形状: {param.shape}")
        #     print(f"  数据类型: {param.dtype}")
        #     print(f"  设备位置: {param.place}")
        #     print(f"  是否可训练: {not param.stop_gradient}")

        parameters = [
            {
                'params': params,
            }
        ]
        super().__init__(learning_rate=lr, parameters=parameters, weight_decay=weight_decay)

        self._hold_steps = hold_steps

    def _on_justify_scale(self, old_scale, new_scale):
        delta = new_scale / old_scale
        for group in self._param_groups:
            for p in group["params"]:
                # print("_on_justify_scale ------", self._accumulators.keys()) 
                if p in self._accumulators:
                    state = self._accumulators[p]
                    if len(state) > 0:
                        if p.dtype == paddle.float16:
                            state["exp_avg"] *= delta
                            state["exp_avg_sq"] *= delta

    @paddle.no_grad()
    def step(self, closure=None, scale=1):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        The remaining arguments are deprecated, and are only retained (for the moment) for error-checking purposes.
        """

        loss = None
        if closure is not None:
            with paddle.enable_grad():
                loss = closure()

        # for idx, group in enumerate(self._param_groups):
        #     print(f"参数组 {idx} 类型: {type(group)}")
        #     if isinstance(group, dict):
        #         print(f"  包含的键: {group.keys()}")
        #         if "params" in group:
        #             params = group["params"]
        #             print(f"  'params' 类型: {type(params)}, 长度: {len(params)}")
        #             for p in params:
        #                 print(f"    参数类型: {type(p)}, 名称: {getattr(p, 'name', '无')}")
        #         else:
        #             print("  错误: 'params' 键缺失！")
        #     else:
        #         print("  错误: 参数组不是字典！")
        # update parameters
        for group in self._param_groups:
            for p in group["params"]:
                if p.grad is not None and not p.stop_gradient:
                    if not p.grad.is_contiguous():
                        p.grad = p.grad.contiguous()
                    # print(f"------p.name----, {p.name}, 原始参数：{p}, 梯度：{p.grad}---")
                    if p.grad.is_sparse():
                        raise RuntimeError(
                            "Adam does not support sparse gradients, please consider SparseAdam instead"
                        )
                    if p.dtype not in [paddle.float32, paddle.float16, paddle.bfloat16]:
                        raise RuntimeError(
                            "Adam only supports fp32, fp16 and bf16 gradients"
                        )

                    state = self._accumulators[p]

                    # Lazy state initialization
                    if len(state) == 0:
                        state["step"] = 0
                        # Exponential moving average of gradient values
                        if p.dtype == paddle.float16:
                            state["exp_avg"] = paddle.zeros(
                                p.shape, dtype=paddle.float16
                            ).cuda()  # on device
                        else:
                            state["exp_avg"] = paddle.zeros(
                                p.shape, dtype=paddle.float32
                            ).cuda()  # on device
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = paddle.zeros(
                            p.shape, dtype=paddle.float32
                        ).cuda()  # on device

                        if p.dtype != paddle.float32:
                            state["_param_fp32"] = paddle.empty(
                                p.shape, dtype=paddle.float32
                            ).cuda() # on device
                            state["_param_fp32"].copy_(p.cast(paddle.float32), True)

                    # update the steps for each param group update
                    if ("maximize" in group) and (group["maximize"] is True):
                        grad = -p.grad
                    else:
                        grad = p.grad

                    if p.dtype == paddle.float32:
                        other_kwargs = {}
                        if (
                            "maximize"
                            in inspect.signature(
                                paddle.optimizer.Adam
                            ).parameters
                        ):
                            other_kwargs["maximize"] = False
                        if "adam_opt" not in state:
                            print(f"参数 p 的名称: {p.name}")
                            print(f"参数 p 的形状: {p.shape}")
                            print(f"参数 p 的维度: {p.ndim}")
                            state["adam_opt"] = paddle.optimizer.Adam(
                                parameters = [p],
                                beta1=self.beta1,
                                beta2=self.beta2,
                                learning_rate=0.0 if state["step"] < self._hold_steps else self.lr,
                                weight_decay=self.weight_decay,
                                epsilon=self.eps
                            )
                        state["adam_opt"].step()
                        state["adam_opt"].clear_grad()
                        state["step"] += 1
                    else:
                        f = F.adam_fp16 if p.dtype == paddle.float16 else F.adam_bf16
                        state["step"] += 1
                        # print("------f(.......)-------")
                        # print("判断参数是否符合预期", state["_param_fp32"].is_contiguous(),
                        #       state["_param_fp32"].place.is_gpu_place(), paddle.CUDAPlace)
                        # print("state['_param_fp32']", state["_param_fp32"].numel(), int(state["_param_fp32"].numel().item()))
                        f(
                            state["_param_fp32"],  # fp32
                            p,  # fp16
                            grad,  # fp16
                            state["exp_avg"],  # fp16: m
                            state["exp_avg_sq"],  # fp32: v
                            self.beta1,
                            self.beta2,
                            self.eps,
                            0.0 if state["step"] < self._hold_steps else self.lr,
                            scale,
                            self.weight_decay,
                            state["step"],
                        )
        # print("adam step ok----------")
        return loss

    def get_avg_delta():

        raise NotImplementedError(
            "get delta info is not supported in Adam optimizer , try bmt.optim.AdamOffloadOptimizer"
        )

    def get_var_delta():

        raise NotImplementedError(
            "get delta info is not supported in Adam optimizer , try bmt.optim.AdamOffloadOptimizer"
        )

    def load_state_dict(self, state_dict: dict) -> None:
        r"""Loads the optimizer state.

        Args:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # deepcopy, to be consistent with module API
        state_dict = deepcopy(state_dict)
        # Validate the state_dict
        groups = self._param_groups
        saved_groups = state_dict["param_groups"]

        if len(groups) != len(saved_groups):
            raise ValueError(
                "loaded state dict has a different number of " "parameter groups"
            )
        param_lens = (len(g["params"]) for g in groups)
        saved_lens = (len(g["params"]) for g in saved_groups)
        if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
            raise ValueError(
                "loaded state dict contains a parameter group "
                "that doesn't match the size of optimizer's group"
            )

        # Update the state
        id_map = {
            old_id: p
            for old_id, p in zip(
                chain.from_iterable((g["params"] for g in saved_groups)),
                chain.from_iterable((g["params"] for g in groups)),
            )
        }

        # Copy state assigned to params (and cast tensors to appropriate types).
        # State that is not assigned to params is copied as is (needed for
        # backward compatibility).
        state = defaultdict(dict)
        for k, v in state_dict["state"].items():
            if k in id_map:
                param = id_map[k]

                if param.dtype != paddle.float32 and "_param_fp32" not in v:
                    v["_param_fp32"] = paddle.empty(
                        param.size(), dtype=paddle.float32, device=param.device
                    )
                    v["_param_fp32"].copy_(param)

                for name, dtype in [
                    (
                        "exp_avg",
                        (
                            paddle.float16
                            if param.dtype == paddle.float16
                            else paddle.float32
                        ),
                    ),
                    ("exp_avg_sq", paddle.float32),
                    ("_param_fp32", paddle.float32),
                ]:
                    if name in v:
                        v[name] = v[name].to(param.device).to(dtype)

                state[param] = v
            else:
                state[k] = v

        # Update parameter groups, setting their 'params' value
        def update_group(group, new_group):
            new_group["params"] = group["params"]
            return new_group

        param_groups = [update_group(g, ng) for g, ng in zip(groups, saved_groups)]
        self.__setstate__({"state": state, "param_groups": param_groups})

    # TODO zero_grad(set_to_none=True) makes optimizer crashed, maybe the reason of grad accu
    def zero_grad(self, set_to_none: bool = False):
        super().zero_grad(set_to_none=set_to_none)
