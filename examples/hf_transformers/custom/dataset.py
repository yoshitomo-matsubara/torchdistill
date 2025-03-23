from transformers import default_data_collator

from torchdistill.common.constant import def_logger
from torchdistill.datasets.registry import register_collate_func

logger = def_logger.getChild(__name__)

register_collate_func(default_data_collator)
