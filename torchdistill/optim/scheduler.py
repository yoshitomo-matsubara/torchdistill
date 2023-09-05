from torch.optim.lr_scheduler import LambdaLR

from .registry import register_scheduler


@register_scheduler
def poly_lr_scheduler(optimizer, num_iterations, num_epochs, power=0.9):
    """
    A "poly" learning rate policy used in `"Rethinking Atrous Convolution for Semantic Image Segmentation" <https://arxiv.org/abs/1706.05587>`_

    .. math:: lr = init\\_lr \\times \\left(1 - \\frac{iter}{num\\_iterations \\times num\\_epochs}\\right)^{power}

    :param optimizer: optimizer.
    :type optimizer: Optimizer
    :param num_iterations: number of iterations per epoch.
    :type num_iterations: int
    :param num_epochs: number of epochs for the training with this scheduler.
    :type num_epochs: int
    :param power: exponent.
    :type power: float
    :return: lambda lr scheduler.
    :rtype: LambdaLR
    """
    lr_scheduler = LambdaLR(optimizer, lambda x: (1 - x / (num_iterations * num_epochs)) ** power)
    return lr_scheduler
