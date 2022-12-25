from . import densenet, resnet, wide_resnet
from ..registry import MODEL_FUNC_DICT

CLASSIFICATION_MODEL_FUNC_DICT = dict()
CLASSIFICATION_MODEL_FUNC_DICT.update(MODEL_FUNC_DICT)
