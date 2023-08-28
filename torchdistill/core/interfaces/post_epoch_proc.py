from torch import distributed as dist
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts

from .registry import register_post_epoch_proc_func
from ...common.constant import def_logger
from ...models.wrapper import AuxiliaryModelWrapper

logger = def_logger.getChild(__name__)


@register_post_epoch_proc_func
def default_post_epoch_process_with_teacher(self, metrics=None, **kwargs):
    """
    Performs post-epoch process for distillation box.

    :param self: distillation box.
    :type self: torchdistill.core.distillation.DistillationBox
    :param metrics: ``metric`` for ReduceLROnPlateau.step.
    :type metrics: Any
    """
    # Epoch-wise scheduler step
    if self.lr_scheduler is not None and self.scheduling_step <= 0:
        if isinstance(self.lr_scheduler, ReduceLROnPlateau):
            self.lr_scheduler.step(metrics)
        elif isinstance(self.lr_scheduler, CosineAnnealingWarmRestarts):
            epoch = self.lr_scheduler.last_epoch + 1
            self.lr_scheduler.step(epoch)
        else:
            self.lr_scheduler.step()
    if isinstance(self.teacher_model, AuxiliaryModelWrapper):
        self.teacher_model.post_epoch_process()
    if isinstance(self.student_model, AuxiliaryModelWrapper):
        self.student_model.post_epoch_process()
    if self.distributed:
        dist.barrier()


@register_post_epoch_proc_func
def default_post_epoch_process_without_teacher(self, metrics=None, **kwargs):
    """
    Performs post-epoch process for training box.

    :param self: training box.
    :type self: torchdistill.core.training.TrainingBox
    :param metrics: ``metric`` for ReduceLROnPlateau.step.
    :type metrics: Any
    """
    # Epoch-wise scheduler step
    if self.lr_scheduler is not None and self.scheduling_step <= 0:
        if isinstance(self.lr_scheduler, ReduceLROnPlateau):
            self.lr_scheduler.step(metrics)
        elif isinstance(self.lr_scheduler, CosineAnnealingWarmRestarts):
            epoch = self.lr_scheduler.last_epoch + 1
            self.lr_scheduler.step(epoch)
        else:
            self.lr_scheduler.step()
    if isinstance(self.model, AuxiliaryModelWrapper):
        self.model.post_epoch_process()
    if self.distributed:
        dist.barrier()
