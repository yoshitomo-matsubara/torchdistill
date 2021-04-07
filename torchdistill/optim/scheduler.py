from torch.optim.lr_scheduler import LambdaLR

from torchdistill.optim.registry import register_scheduler


@register_scheduler
def poly_lr_scheduler(optimizer, num_iterations, num_epochs, power=0.9):
    lr_scheduler = LambdaLR(optimizer, lambda x: (1 - x / (num_iterations * num_epochs)) ** power)
    return lr_scheduler
