import os

import torch
from torchdistill.common.constant import def_logger
from torchdistill.common.module_util import check_if_wrapped

logger = def_logger.getChild(__name__)


def save_ckpt(model, tokenizer, optimizer, scheduler, config, args, ckpt_dir_path):
    if not os.path.exists(ckpt_dir_path):
        os.makedirs(ckpt_dir_path)

    model_to_save = model.module if check_if_wrapped(model) else model
    logger.info(f'Saving model checkpoint to {ckpt_dir_path}')
    model_to_save.save_pretrained(ckpt_dir_path)
    tokenizer.save_pretrained(ckpt_dir_path)
    torch.save(args, os.path.join(ckpt_dir_path, 'training_args.bin'))
    torch.save(config, os.path.join(ckpt_dir_path, 'config.bin'))
    logger.info(f'Saving optimizer and scheduler states to {ckpt_dir_path}')
    torch.save(optimizer.state_dict(), os.path.join(ckpt_dir_path, 'optimizer.pt'))
    torch.save(scheduler.state_dict(), os.path.join(ckpt_dir_path, 'scheduler.pt'))
