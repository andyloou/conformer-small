"""Warm up learning rate scheduler module."""
from typing import Union

import torch
from torch.optim.lr_scheduler import _LRScheduler


class NoamLR(_LRScheduler):
    """

     NoamLR Scheduler:

    NoamLR:
        lr = optimizer.lr * model_size ** -0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: Union[int, float] = 25000,
        d_model: int  = 176,
        last_epoch: int = -1,
        
    ):
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        # __init__() must be invoked before setting field
        # because step() is also invoked in __init__()
        super().__init__(optimizer, last_epoch)

    def __repr__(self):
        return f"{self.__class__.__name__}(warmup_steps={self.warmup_steps})"

    def get_lr(self):
        step_num = max(1, self.last_epoch + 1)
        scale = self.d_model ** -0.5
        factor = min(step_num**-0.5, step_num * self.warmup_steps**-1.5)
        return [scale * factor * base_lr for base_lr in self.base_lrs]
