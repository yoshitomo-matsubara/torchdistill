from torchdistill.models.classification import densenet, resnet, wide_resnet
from torchdistill.models.registry import MODEL_FUNC_DICT

CLASSIFICATION_MODEL_FUNC_DICT = dict()
CLASSIFICATION_MODEL_FUNC_DICT.update(MODEL_FUNC_DICT)
