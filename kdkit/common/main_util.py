import logging
import os

import torch
import torch.distributed as dist

from kdkit.common.constant import def_logger
from myutils.common.file_util import check_if_exists, make_parent_dirs
from myutils.pytorch.module_util import check_if_wrapped

logger = def_logger.getChild(__name__)


def setup_for_distributed(is_master):
    """
    This function disables logging when not in master process
    """
    def_logger.setLevel(logging.INFO if is_master else logging.WARN)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(world_size=1, dist_url='env://'):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        device_id = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        device_id = rank % torch.cuda.device_count()
    else:
        logger.info('Not using distributed mode')
        return False, None

    torch.cuda.set_device(device_id)
    dist_backend = 'nccl'
    logger.info('| distributed init (rank {}): {}'.format(rank, dist_url))
    torch.distributed.init_process_group(backend=dist_backend, init_method=dist_url,
                                         world_size=world_size, rank=rank)
    torch.distributed.barrier()
    setup_for_distributed(rank == 0)
    return True, [device_id]


def load_ckpt(ckpt_file_path, model=None, optimizer=None, lr_scheduler=None, strict=True):
    if not check_if_exists(ckpt_file_path):
        logger.info('ckpt file is not found at `{}`'.format(ckpt_file_path))
        return None, None

    ckpt = torch.load(ckpt_file_path, map_location='cpu')
    if model is not None:
        logger.info('Loading model parameters')
        model.load_state_dict(ckpt['model'], strict=strict)
    if optimizer is not None:
        logger.info('Loading optimizer parameters')
        optimizer.load_state_dict(ckpt['optimizer'])
    if lr_scheduler is not None:
        logger.info('Loading scheduler parameters')
        lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
    return ckpt.get('best_value', 0.0), ckpt.get('config', None), ckpt.get('args', None)


def save_ckpt(model, optimizer, lr_scheduler, best_value, config, args, output_file_path):
    make_parent_dirs(output_file_path)
    model_state_dict = model.module.state_dict() if check_if_wrapped(model) else model.state_dict()
    lr_scheduler_state_dict = lr_scheduler.state_dict() if lr_scheduler is not None else None
    save_on_master({'model': model_state_dict, 'optimizer': optimizer.state_dict(), 'best_value': best_value,
                    'lr_scheduler': lr_scheduler_state_dict, 'config': config, 'args': args}, output_file_path)
