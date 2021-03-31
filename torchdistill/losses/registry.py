from torchdistill.common import misc_util

LOSS_DICT = misc_util.get_classes_as_dict('torch.nn.modules.loss')


def get_loss(loss_type, param_dict=None, **kwargs):
    if param_dict is None:
        param_dict = dict()
    lower_loss_type = loss_type.lower()
    if lower_loss_type in LOSS_DICT:
        return LOSS_DICT[lower_loss_type](**param_dict, **kwargs)
    raise ValueError('loss_type `{}` is not expected'.format(loss_type))
