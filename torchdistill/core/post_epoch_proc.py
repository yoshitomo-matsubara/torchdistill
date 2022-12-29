from torch import distributed as dist
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR

from .registry import register_post_epoch_proc_func
from ..common.constant import def_logger
from ..models.special import AuxiliaryModelWrapper

logger = def_logger.getChild(__name__)


@register_post_epoch_proc_func
def default_post_epoch_process_with_teacher(self, **kwargs):
    # Epoch-wise scheduler step
    if self.lr_scheduler is not None and self.scheduling_step <= 0:
        if isinstance(self.lr_scheduler, ReduceLROnPlateau):
            metrics = kwargs['metrics']
            self.lr_scheduler.step(metrics)
        elif isinstance(self.lr_scheduler, LambdaLR):
            epoch = self.lr_scheduler.last_epoch + 1
            self.lr_scheduler.step(epoch)
        else:
            self.lr_scheduler.step()
    if isinstance(self.teacher_model, AuxiliaryModelWrapper):
        self.teacher_model.post_process()
    if isinstance(self.student_model, AuxiliaryModelWrapper):
        self.student_model.post_process()
    if self.distributed:
        dist.barrier()


@register_post_epoch_proc_func
def default_post_epoch_process_without_teacher(self, **kwargs):
    # Epoch-wise scheduler step
    if self.lr_scheduler is not None and self.scheduling_step <= 0:
        if isinstance(self.lr_scheduler, ReduceLROnPlateau):
            metrics = kwargs['metrics']
            self.lr_scheduler.step(metrics)
        elif isinstance(self.lr_scheduler, LambdaLR):
            epoch = self.lr_scheduler.last_epoch + 1
            self.lr_scheduler.step(epoch)
        else:
            self.lr_scheduler.step()
    if isinstance(self.model, AuxiliaryModelWrapper):
        self.model.post_process()
    if self.distributed:
        dist.barrier()
