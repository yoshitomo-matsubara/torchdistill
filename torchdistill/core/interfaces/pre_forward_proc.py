from .registry import register_pre_forward_proc_func
from ...common.constant import def_logger

logger = def_logger.getChild(__name__)


@register_pre_forward_proc_func
def default_pre_forward_process(self, *args, **kwargs):
    pass
