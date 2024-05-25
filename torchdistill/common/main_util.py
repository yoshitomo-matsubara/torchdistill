import builtins as __builtin__
import logging
import os
import random
from importlib import import_module

import numpy as np
import torch
import torch.distributed as dist

from .constant import def_logger
from .file_util import check_if_exists, make_parent_dirs
from .module_util import check_if_wrapped

logger = def_logger.getChild(__name__)


def import_dependencies(dependencies=None):
    """
    Imports specified packages.

    :param dependencies: package names.
    :type dependencies: list[dict or list[str] or (str, str) or str] or str or None
    """
    if dependencies is None:
        return

    if isinstance(dependencies, str):
        dependencies = [dependencies]

    for dependency in dependencies:
        name = None
        package = None
        if isinstance(dependency, dict):
            import_module(**dependency)
            name = dependency.get('name', None)
            package = dependency.get('package', None)
        elif isinstance(dependency, (list, tuple)):
            import_module(*dependency)
            name = dependency[0]
            if len(dependency) >= 2:
                package = dependency[1]
        elif isinstance(dependency, str):
            import_module(dependency)
            package = dependency
        else:
            raise TypeError(f'Failed to import module with `{dependency}`')
        if package is None:
            logger.info(f'Imported `{name}`')
        else:
            logger.info(f'Imported `{name}` from `{package}`')


def import_get(key, package=None, **kwargs):
    """
    Imports module and get its attribute.

    :param key: attribute name or package path separated by period(.).
    :type key: str
    :param package: package path if ``key`` is just an attribute name.
    :type package: str or None
    :return: attribute of the imported module.
    :rtype: Any
    """
    if package is None:
        names = key.split('.')
        key = names[-1]
        package = '.'.join(names[:-1])

    logger.info(f'Getting `{key}` from `{package}`')
    module = import_module(package)
    return getattr(module, key)


def import_call(key, package=None, init=None, **kwargs):
    """
    Imports module and call the module/function e.g., instantiation.

    :param key: module name or package path separated by period(.).
    :type key: str
    :param package: package path if ``key`` is just an attribute name.
    :type package: str or None
    :param init: dict of arguments and/or keyword arguments to instantiate the imported module.
    :type init: dict
    :return: object imported and called.
    :rtype: Any
    """
    if package is None:
        names = key.split('.')
        key = names[-1]
        package = '.'.join(names[:-1])

    obj = import_get(key, package)
    if init is None:
        init = dict()

    logger.info(f'Calling `{key}` from `{package}` with {init}')
    args = init.get('args', list())
    kwargs = init.get('kwargs', dict())
    return obj(*args, **kwargs)


def import_call_method(package, class_name=None, method_name=None, init=None, **kwargs):
    """
    Imports module and call its method.

    :param package: package path.
    :type package: str
    :param class_name: class name under ``package``.
    :type class_name: str
    :param method_name: method name of ``class_name`` class under ``package``.
    :type method_name: str
    :param init: dict of arguments and/or keyword arguments to instantiate the imported module.
    :type init: dict
    :return: object imported and called.
    :rtype: Any
    """
    if class_name is None or method_name is None:
        names = package.split('.')
        class_name = names[-2]
        method_name = names[-1]
        package = '.'.join(names[:-2])

    cls = import_get(class_name, package)
    if init is None:
        init = dict()

    logger.info(f'Calling `{class_name}.{method_name}` from `{package}` with {init}')
    args = init.get('args', list())
    kwargs = init.get('kwargs', dict())
    method = getattr(cls, method_name)
    return method(*args, **kwargs)


def setup_for_distributed(is_master):
    """
    Disables logging when not in master process.

    :param is_master: True if it is the master process.
    :type is_master: bool
    """
    def_logger.setLevel(logging.INFO if is_master else logging.WARN)
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def set_seed(seed):
    """
    Sets a random seed for `random`, `numpy`, and `torch` (torch.manual_seed, torch.cuda.manual_seed_all).

    :param seed: random seed.
    :type seed: int
    """
    if not isinstance(seed, int):
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_dist_avail_and_initialized():
    """
    Checks if distributed model is available and initialized.

    :return: True if distributed mode is available and initialized.
    :rtype: bool
    """
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    """
    Gets world size.

    :return: world size.
    :rtype: int
    """
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """
    Gets the rank of the current process in the provided ``group`` or the default group if none was provided.

    :return: rank of the current process in the provided ``group`` or the default group if none was provided.
    :rtype: int
    """
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    """
    Checks if this is the main process.

    :return: True if this is the main process.
    :rtype: bool
    """
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    """
    Use `torch.save` for `args` if this is the main process.

    :return: True if this is the main process.
    :rtype: bool
    """
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(world_size=1, dist_url='env://'):
    """
    Initialize the distributed mode.

    :param world_size: world size.
    :type world_size: int
    :param dist_url: URL specifying how to initialize the process group.
    :type dist_url: str
    :return: tuple of 1) whether distributed mode is initialized, 2) world size, and 3) list of device IDs.
    :rtype: (bool, int, list[int] or None)
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        device_id = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        device_id = rank % torch.cuda.device_count()
    else:
        logger.info('Not using distributed mode')
        return False, world_size, None

    torch.cuda.set_device(device_id)
    dist_backend = 'nccl'
    logger.info('| distributed init (rank {}): {}'.format(rank, dist_url))
    torch.distributed.init_process_group(backend=dist_backend, init_method=dist_url,
                                         world_size=world_size, rank=rank)
    torch.distributed.barrier()
    setup_for_distributed(rank == 0)
    return True, world_size, [device_id]


def load_ckpt(ckpt_file_path, model=None, optimizer=None, lr_scheduler=None, strict=True):
    """
    Load a checkpoint file with model, optimizer, and/or lr_scheduler.

    :param ckpt_file_path: checkpoint file path.
    :type ckpt_file_path: str
    :param model: model.
    :type model: nn.Module
    :param optimizer: optimizer.
    :type optimizer: nn.Module
    :param lr_scheduler: learning rate scheduler.
    :type lr_scheduler: nn.Module
    :param strict: ``strict`` as a keyword argument of ``load_state_dict``.
    :type strict: bool
    :return: tuple of best value (e.g., best validation result) and parsed args.
    :rtype: (float or None, argparse.Namespace or None)
    """
    if check_if_exists(ckpt_file_path):
        ckpt = torch.load(ckpt_file_path, map_location='cpu')
    elif isinstance(ckpt_file_path, str) and \
            (ckpt_file_path.startswith('https://') or ckpt_file_path.startswith('http://')):
        ckpt = torch.hub.load_state_dict_from_url(ckpt_file_path, map_location='cpu', progress=True)
    else:
        message = 'ckpt file path is None' if ckpt_file_path is None \
            else 'ckpt file is not found at `{}`'.format(ckpt_file_path)
        logger.info(message)
        return None, None

    if model is not None:
        if 'model' in ckpt:
            logger.info('Loading model parameters')
            if strict is None:
                model.load_state_dict(ckpt['model'], strict=strict)
            else:
                model.load_state_dict(ckpt['model'], strict=strict)
        elif optimizer is None and lr_scheduler is None:
            logger.info('Loading model parameters only')
            model.load_state_dict(ckpt, strict=strict)
        else:
            logger.warning('No model parameters found')

    if optimizer is not None:
        if 'optimizer' in ckpt:
            logger.info('Loading optimizer parameters')
            optimizer.load_state_dict(ckpt['optimizer'])
        elif model is None and lr_scheduler is None:
            logger.info('Loading optimizer parameters only')
            optimizer.load_state_dict(ckpt)
        else:
            logger.warning('No optimizer parameters found')

    if lr_scheduler is not None:
        if 'lr_scheduler' in ckpt:
            logger.info('Loading scheduler parameters')
            lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        elif model is None and optimizer is None:
            logger.info('Loading scheduler parameters only')
            lr_scheduler.load_state_dict(ckpt)
        else:
            logger.warning('No scheduler parameters found')
    return ckpt.get('best_value', 0.0), ckpt.get('args', None)


def save_ckpt(model, optimizer, lr_scheduler, best_value, args, output_file_path):
    """
    Save a checkpoint file including model, optimizer, best value, parsed args, and learning rate scheduler.

    :param model: model.
    :type model: nn.Module
    :param optimizer: optimizer.
    :type optimizer: nn.Module
    :param lr_scheduler: learning rate scheduler.
    :type lr_scheduler: nn.Module
    :param best_value: best value e.g., best validation result.
    :type best_value: float
    :param args: parsed args.
    :type args: argparse.Namespace
    :param output_file_path: output file path.
    :type output_file_path: str
    """
    make_parent_dirs(output_file_path)
    model_state_dict = model.module.state_dict() if check_if_wrapped(model) else model.state_dict()
    lr_scheduler_state_dict = lr_scheduler.state_dict() if lr_scheduler is not None else None
    save_on_master({'model': model_state_dict, 'optimizer': optimizer.state_dict(), 'best_value': best_value,
                    'lr_scheduler': lr_scheduler_state_dict, 'args': args}, output_file_path)
