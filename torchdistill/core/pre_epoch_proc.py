from .registry import register_pre_epoch_proc_func
from .util import clear_io_dict
from ..common.constant import def_logger

logger = def_logger.getChild(__name__)


@register_pre_epoch_proc_func
def default_pre_epoch_process_with_teacher(self, epoch, **kwargs):
    clear_io_dict(self.teacher_io_dict)
    clear_io_dict(self.student_io_dict)
    self.teacher_model.eval()
    self.student_model.train()
    if self.distributed:
        self.train_data_loader.batch_sampler.sampler.set_epoch(epoch)


@register_pre_epoch_proc_func
def default_pre_epoch_process_without_teacher(self, epoch, **kwargs):
    clear_io_dict(self.model_io_dict)
    self.model.train()
    if self.distributed:
        self.train_data_loader.batch_sampler.sampler.set_epoch(epoch)
