from torchdistill.models.adaptation import ADAPTATION_CLASS_DICT
from torchdistill.models.custom import CUSTOM_MODEL_CLASS_DICT, CUSTOM_MODEL_FUNC_DICT
from torchdistill.models.special import SPECIAL_CLASS_DICT

MODEL_DICT = dict()

MODEL_DICT.update(ADAPTATION_CLASS_DICT)
MODEL_DICT.update(SPECIAL_CLASS_DICT)
MODEL_DICT.update(CUSTOM_MODEL_CLASS_DICT)
MODEL_DICT.update(CUSTOM_MODEL_FUNC_DICT)
