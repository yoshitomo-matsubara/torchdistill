from torch.utils.data.distributed import DistributedSampler

from .registry import register_pre_epoch_proc_func
from ..util import clear_io_dict
from ...common.constant import def_logger

logger = def_logger.getChild(__name__)


def _set_epoch(data_loader, epoch):
    # Accelerate's prepared loaders wrap the batch sampler (e.g., BatchSamplerShard has no `sampler`)
    # and reshuffle across processes in their own set_epoch
    if hasattr(data_loader, 'set_epoch'):
        data_loader.set_epoch(epoch)
        return

    sampler = getattr(data_loader.batch_sampler, 'sampler', None)
    if isinstance(sampler, DistributedSampler):
        sampler.set_epoch(epoch)


@register_pre_epoch_proc_func
def default_pre_epoch_process_with_teacher(self, epoch=None, **kwargs):
    """
    Performs pre-epoch process for distillation box.

    :param self: distillation box.
    :type self: torchdistill.core.distillation.DistillationBox or torchdistill.core.training.TrainingBox
    :param epoch: ``epoch`` for DistributedSampler.
    :type epoch: int
    """
    clear_io_dict(self.teacher_io_dict)
    clear_io_dict(self.student_io_dict)
    self.student_model.train()
    if self.distributed:
        _set_epoch(self.train_data_loader, epoch)


@register_pre_epoch_proc_func
def default_pre_epoch_process_without_teacher(self, epoch=None, **kwargs):
    """
    Performs pre-epoch process for training box.

    :param self: distillation box.
    :type self: torchdistill.core.distillation.DistillationBox or torchdistill.core.training.TrainingBox
    :param epoch: ``epoch`` for DistributedSampler.
    :type epoch: int
    """
    clear_io_dict(self.model_io_dict)
    self.model.train()
    if self.distributed:
        _set_epoch(self.train_data_loader, epoch)
