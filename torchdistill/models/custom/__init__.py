from torchdistill.models.custom import bottleneck
from torchdistill.models.registry import MODEL_CLASS_DICT, MODEL_FUNC_DICT

CUSTOM_MODEL_CLASS_DICT = dict()
CUSTOM_MODEL_FUNC_DICT = dict()

CUSTOM_MODEL_CLASS_DICT.update(MODEL_CLASS_DICT)
CUSTOM_MODEL_FUNC_DICT.update(MODEL_FUNC_DICT)
