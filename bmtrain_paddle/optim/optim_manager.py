from typing import Optional, Union, List, Dict, Tuple
import paddle
from ..loss._function import has_inf_nan
from ..utils import print_rank
from ..lr_scheduler.warmup import WarmupLRScheduler
from .. import nccl
from ..global_var import config

def check_overflow(param_groups):
    # check overflow
    has_inf_or_nan = paddle.zeros(1, dtype=paddle.uint8).cuda()
    for group in param_groups:
        for p in group['params']:
            if p.grad is not None:
                if p.dtype != paddle.float32:
                    # print("------check_overflow parameter name:--------", p.name)
                    has_inf_nan(p.grad, has_inf_or_nan)
    if "comm" in config:
        nccl.allReduce(has_inf_or_nan, has_inf_or_nan, "max", config["comm"])

    # print(f"check_overflow {has_inf_or_nan.item()}")
    if has_inf_or_nan > 0:
        raise OverflowError("Gradient overflow")

def grad_rescale(param_groups, scale):
    for group in param_groups:
        for p in group['params']:
            if p.grad is not None and not p.stop_gradient:
                p.grad /= scale

class OptimManager:
    """wait cuda stream. Optional: add loss scaler for mix-precision training

    Args:
        loss_scale (float): The initial loss scale. Default to None for not using loss scaling.
        loss_scale_factor (float): The loss scale factor.
        loss_scale_steps (int): The loss scale steps.

    Examples:
        >>> optim_manager = bmt.optim.OptimManager(loss_scale=1024)
        >>> optim_manager.add_optimizer(optimizer1)
        >>> optim_manager.add_optimizer(optimizer2, lr_scheduler2)
        >>> for data in dataset:
        >>>     # forward pass and calculate loss
        >>>     optim_manager.zero_grad()
        >>>     optim_manager.backward(loss)
        >>>     optim_manager.clip_grad_norm(optimizer1.param_groups, max_norm=1.0, norm_type=2)
        >>>     optim_manager.clip_grad_norm(optimizer2.param_groups, max_norm=2.0, norm_type=2)
        >>>     optim_manager.step()
    """
    def __init__(self,
        loss_scale : Optional[float] = None,
        loss_scale_factor : float = 2,
        loss_scale_steps : int = 1024,
        min_loss_scale = 1,
        max_loss_scale = float("inf"),
        grad_scale : Optional[int] = None,
    ):
        if loss_scale is not None:
            self.loss_scale = loss_scale
            self.loss_scale_enabled = True
        else:
            self.loss_scale = 1
            self.loss_scale_enabled = False
        self.steps_since_last_scale = 0
        self.loss_scale_factor = loss_scale_factor if loss_scale_factor > 1 else 1 / loss_scale_factor
        self.loss_scale_steps = loss_scale_steps
        self.min_loss_scale = min_loss_scale
        self.max_loss_scale = max_loss_scale
        if grad_scale is None:
            grad_scale = config['zero_size']
        self.grad_scale = grad_scale

        self.optimizers = []
        self.lr_schedulers = []

    def add_optimizer(
        self,
        optimizer: paddle.optimizer.Optimizer,
        lr_scheduler: Optional[WarmupLRScheduler] = None,
    ):
        """Add optimizer and (optional) its corresponding lr_scheduler into optim_manager.
        All optimizers in the same optim_manager share the same loss scale.

        Args:
            optim (torch.optim.Optimizer): A pytorch optimizer, e.g. torch.optim.Adam, torch.optim.SGD or bmtrain.optim.AdamOffloadOptimizer
            lr_scheduler (Optional[WarmupLRScheduler]): A warmup lr scheduler, e.g. bmt.lr_scheduler.Noam
        """
        self.optimizers.append(optimizer)
        self.lr_schedulers.append(lr_scheduler)

    def scale_loss(self, loss : paddle.Tensor) -> paddle.Tensor:
        res = loss * ( self.loss_scale / self.grad_scale ) # loss scale
        return res

    def backward(self, loss : paddle.Tensor):
        """
        Backward with loss scale.

        Args:
            loss (torch.Tensor): loss
        """
        loss = self.scale_loss(loss)
        loss.stop_gradient = False
        print("--------------loss backward--------------", loss)
        loss.backward()
        # some reduce ops of distributed parameter were launched on load stream
        current_stream = paddle.device.cuda.current_stream()
        current_stream.wait_stream(config['load_stream'])

    def zero_grad(self):
        """
        This is a helper function to call optimizer.zero_grad()
        """
        for optimizer in self.optimizers:
            # optimizer.clear_grad(set_to_none=False)
            optimizer.clear_grad()

    def step(self):
        """
        Backward with loss scale.
        Synchronize streams before optimizer steps.

        This is a helper function to call optimizer.step() and lr_scheduler.step() and synchronize streams.

        This function can also handle gradient overflow by reducing the loss scale when it occurs.
        """
        if self.loss_scale_enabled:
            has_overflow = False
            for optimizer in self.optimizers:
                # print_rank(f"\n[Debug] Checking optimizer: {type(optimizer).__name__}")
                # print("[Debug] Type of optimizer._param_groups:", type(optimizer._param_groups))
                # print("[Debug] Length of optimizer._param_groups:", len(optimizer._param_groups))
                # for group_idx, param_group in enumerate(optimizer._param_groups):
                #     # 打印参数组元信息
                #     print_rank(f"---- Parameter Group {group_idx} {param_group}----")
                #     print_rank("Group Hyperparameters:")
                #     for key in param_group:
                #         if key != "params":
                #             print_rank(f"  {key}: {param_group[key]}")

                #     # 打印参数详细信息
                #     print_rank("Parameters in group:")
                #     for param_idx, param in enumerate(param_group["params"]):
                #         param_info = [
                #             f"  Param {param_idx}",
                #             f"Name: {param.name}" if hasattr(param, 'name') else "",
                #             f"Shape: {param.shape}",
                #             f"Dtype: {param.dtype}",
                #             f"Requires_grad: {not param.stop_gradient}",
                #             f"Grad: {'Exists' if param.grad is not None else 'None'}"
                #         ]
                #         print_rank("\n".join(param_info))
                        
                #         # 可选：打印梯度统计信息
                #         if param.grad is not None:
                #             grad = param.grad
                #             print_rank(f"  Grad Statistics: mean={grad.mean().item()}, std={grad.std().item()}, min={grad.min().item()}, max={grad.max().item()}")
                try:
                    check_overflow(optimizer._param_groups)
                except OverflowError:
                    has_overflow = True
                    break
            if has_overflow:
                print_rank("Gradient overflow, change scale from %lf to %lf" % (self.loss_scale, self.loss_scale / self.loss_scale_factor))
                with paddle.no_grad():
                    if self.loss_scale > self.min_loss_scale:
                        self._justify_scale(self.loss_scale / self.loss_scale_factor)
                    self.zero_grad()
                return
        for optimizer, lr_scheduler in zip(self.optimizers, self.lr_schedulers):
            if hasattr(optimizer, "_bmtrain_optimizer") and optimizer._bmtrain_optimizer:
                print("optimizer.step(scale=self.loss_scale)")
                optimizer.step(scale=self.loss_scale)
            else:
                if self.loss_scale_enabled:
                    grad_rescale(optimizer._param_groups, self.loss_scale)
                print("optimizer.step()")
                optimizer.step()

            if lr_scheduler is not None:
                print("---------lr_scheduler.step()-----")
                lr_scheduler.step()

        if self.loss_scale_enabled:
            self.steps_since_last_scale += 1

            if self.steps_since_last_scale >= self.loss_scale_steps and self.loss_scale < self.max_loss_scale:
                self._justify_scale(self.loss_scale * self.loss_scale_factor)

        current_stream = paddle.device.cuda.current_stream()
        config['load_stream'].wait_stream(current_stream)

    def clip_grad_norm(self, param_groups, max_norm, norm_type=2, eps=1e-6):
        """Clips gradient norm of an iterable of parameters.

        The norm is computed over all gradients together, as if they were concatenated into a single vector. Gradients are modified in-place.

        Args:
            parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a single Tensor that will have gradients normalized.
            max_norm (float or int): max norm of the gradients.
            norm_type (float or int): type of the used p-norm. Can be 'inf' for infinity norm.
            eps (float): epsilon used to avoid zero division.

        Returns:
            Total norm of the parameters (viewed as a single vector).
        """
        scale = self.loss_scale
        grads = []
        parameters = [p for group in param_groups for p in group['params']]
        for p in parameters:
            if p.grad is not None:
                grads.append(p.grad.data)
            else:
                grads.append(paddle.zeros_like(p.data))

        if norm_type == 'inf':
            total_norm_cuda = max(g.data.abs().max() for g in grads).detach()
            nccl.allReduce(total_norm_cuda.storage(), total_norm_cuda.storage(), "max", config["comm"])
            total_norm = total_norm_cuda
        else:
            norm_type = float(norm_type)
            total_norm_cuda = paddle.device.cuda.FloatTensor([0])
            for index, g in enumerate(grads):
                param_norm = g.data.float().norm(norm_type)
                total_norm_cuda += param_norm ** norm_type
            nccl.allReduce(total_norm_cuda.storage(), total_norm_cuda.storage(), "sum", config["comm"])
            total_norm = total_norm_cuda[0] ** (1. / norm_type)
        # total_norm = total_norm / scale
        # clip_coef = float(max_norm) / (total_norm + eps)
        clip_coef = float(max_norm * scale) / (total_norm + eps)
        if clip_coef < 1:
            for p in parameters:
                if p.grad is not None:
                    p.grad.data.mul_(clip_coef)
        return total_norm / scale

    @paddle.no_grad()
    def _justify_scale(self, scale):
        for optimizer in self.optimizers:
            if hasattr(optimizer, "_on_justify_scale"):
                optimizer._on_justify_scale(self.loss_scale, scale)
        self.loss_scale = scale
        self.steps_since_last_scale = 0

    def state_dict(self, gather_opt=False) -> dict:
        return {
            "optimizers": [opt.state_dict(gather_opt) for opt in self.optimizers],
            "lr_schedulers": [lrs.state_dict() if lrs else None for lrs in self.lr_schedulers],
            "loss_scale": self.loss_scale,
            "loss_scale_enabled": self.loss_scale_enabled,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        assert len(self.optimizers) == len(state_dict["optimizers"])
        assert len(self.lr_schedulers) == len(state_dict["lr_schedulers"])
        for opt, opt_st in zip(self.optimizers, state_dict["optimizers"]):
            opt.load_state_dict(opt_st)
        for lrs, lrs_st in zip(self.lr_schedulers, state_dict["lr_schedulers"]):
            lrs.load_state_dict(lrs_st)
        self.loss_scale = state_dict["loss_scale"]
        self.loss_scale_enabled = state_dict["loss_scale_enabled"]
