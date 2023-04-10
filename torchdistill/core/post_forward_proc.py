import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR, CosineAnnealingWarmRestarts

from .registry import register_post_forward_proc_func
from ..common.constant import def_logger

logger = def_logger.getChild(__name__)


@register_post_forward_proc_func
def default_post_forward_process(self, loss, metrics=None, **kwargs):
    self.stage_grad_count += 1
    if self.grad_accum_step > 1:
        loss /= self.grad_accum_step

    if self.accelerator is not None:
        self.accelerator.backward(loss)
    else:
        loss.backward()

    if self.stage_grad_count % self.grad_accum_step == 0:
        if self.max_grad_norm is not None:
            target_params = [p for group in self.optimizer.param_groups for p in group['kwargs']]
            torch.nn.utils.clip_grad_norm_(target_params, self.max_grad_norm)

        self.optimizer.step()
        self.optimizer.zero_grad()

    # Step-wise scheduler step
    if self.lr_scheduler is not None and self.scheduling_step > 0 \
            and self.stage_grad_count % self.scheduling_step == 0:
        if isinstance(self.lr_scheduler, ReduceLROnPlateau):
            self.lr_scheduler.step(metrics)
        elif isinstance(self.lr_scheduler, (LambdaLR, CosineAnnealingWarmRestarts)):
            local_epoch = int(self.stage_grad_count / self.scheduling_step)
            self.lr_scheduler.step(local_epoch)
        else:
            self.lr_scheduler.step()
