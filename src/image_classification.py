import argparse
import datetime
import time

import torch
import torchvision
from torch import distributed as dist
from torch.backends import cudnn
from torch.nn import DataParallel, SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel

from misc.log import SmoothedValue, MetricLogger
from myutils.common import file_util, yaml_util
from myutils.pytorch import module_util
from tools.distillation import get_distillation_box
from utils import dataset_util, main_util
from utils.eval_util import compute_accuracy


def get_argparser():
    parser = argparse.ArgumentParser(description='Knowledge distillation for image classification models')
    parser.add_argument('--config', required=True, help='yaml file path')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('-use_cache', action='store_true',
                        help='Cache the datasets for quicker initialization. It also serializes the transforms')
    parser.add_argument('-sync_bn', action='store_true', help='Use sync batch norm')
    parser.add_argument('-test_only', action='store_true', help='Only test the models')
    parser.add_argument('-student_only', action='store_true', help='Test the student model only')
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def load_ckpt(ckpt_file_path, model=None, optimizer=None, lr_scheduler=None, strict=True):
    if not file_util.check_if_exists(ckpt_file_path):
        print('ckpt file is not found at `{}`'.format(ckpt_file_path))
        return None, None

    ckpt = torch.load(ckpt_file_path, map_location='cpu')
    if model is not None:
        print('Loading model parameters')
        model.load_state_dict(ckpt['model'], strict=strict)
    if optimizer is not None:
        print('Loading optimizer parameters')
        optimizer.load_state_dict(ckpt['optimizer'])
    if lr_scheduler is not None:
        print('Loading scheduler parameters')
        lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
    return ckpt.get('best_value', 0.0), ckpt['config'], ckpt['args']


def get_model(model_config, device, distributed, sync_bn):
    model_name = model_config['name']
    model = torchvision.models.__dict__[model_name](**model_config['params'])
    if distributed and sync_bn:
        model = SyncBatchNorm.convert_sync_batchnorm(model)

    ckpt_file_path = model_config['ckpt']
    load_ckpt(ckpt_file_path, model=model, strict=True)
    return model.to(device)


def save_ckpt(model, optimizer, lr_scheduler, best_value, config, args, output_file_path):
    file_util.make_parent_dirs(output_file_path)
    model_state_dict =\
        model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()
    main_util.save_on_master({'model': model_state_dict, 'optimizer': optimizer.state_dict(), 'best_value': best_value,
                              'lr_scheduler': lr_scheduler.state_dict(), 'config': config, 'args': args},
                             output_file_path)


def distill_one_epoch(distillation_box, device, epoch, log_freq):
    metric_logger = MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', SmoothedValue(window_size=10, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    for sample_batch, targets in metric_logger.log_every(distillation_box.train_data_loader, log_freq, header):
        start_time = time.time()
        sample_batch, targets = sample_batch.to(device), targets.to(device)
        loss = distillation_box(sample_batch, targets)
        distillation_box.update_params(loss)
        batch_size = sample_batch.shape[0]
        metric_logger.update(loss=loss.item(), lr=distillation_box.optimizer.param_groups[0]['lr'])
        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))


@torch.no_grad()
def evaluate(model, data_loader, device, log_freq=1000, title=None):
    if title is not None:
        print(title)

    num_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    model.eval()
    metric_logger = MetricLogger(delimiter='  ')
    header = 'Test:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, log_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)

            acc1, acc5 = compute_accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    top1_accuracy = metric_logger.acc1.global_avg
    top5_accuracy = metric_logger.acc5.global_avg
    print(' * Acc@1 {:.4f}\tAcc@5 {:.4f}\n'.format(top1_accuracy, top5_accuracy))
    torch.set_num_threads(num_threads)
    return metric_logger.acc1.global_avg


def distill(teacher_model, student_model, dataset_dict, device, distributed, start_epoch, config, args):
    print('Start knowledge distillation')
    train_config = config['train']
    distillation_box = get_distillation_box(teacher_model, student_model, dataset_dict, train_config, distributed)
    ckpt_file_path = config['student_model']['ckpt']
    best_val_top1_accuracy = 0.0
    optimizer, lr_scheduler = distillation_box.optimizer, distillation_box.lr_scheduler
    if file_util.check_if_exists(ckpt_file_path):
        best_val_map, _, _ =\
            load_ckpt(ckpt_file_path, optimizer=optimizer, lr_scheduler=lr_scheduler)

    log_freq = train_config['log_freq']
    student_model_without_ddp = \
        student_model.module if isinstance(student_model, DistributedDataParallel) else student_model
    start_time = time.time()
    for epoch in range(start_epoch, train_config['num_epochs']):
        distillation_box.pre_process(epoch=epoch)
        teacher_model.eval()
        student_model.train()
        distill_one_epoch(distillation_box, device, epoch, log_freq)
        val_top1_accuracy = evaluate(student_model, distillation_box.val_data_loader, device=device, log_freq=log_freq)
        if val_top1_accuracy > best_val_top1_accuracy and main_util.is_main_process():
            print('Updating ckpt (Best top1 accuracy: {:.4f} -> {:.4f})'.format(best_val_top1_accuracy,
                                                                                val_top1_accuracy))
            best_val_top1_accuracy = val_top1_accuracy
            save_ckpt(student_model_without_ddp, optimizer, lr_scheduler,
                      best_val_top1_accuracy, config, args, ckpt_file_path)
        distillation_box.post_process(epoch=epoch)

    dist.barrier()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    distillation_box.clean_modules()


def main(args):
    distributed, device_ids = main_util.init_distributed_mode(args.world_size, args.dist_url)
    print(args)
    cudnn.benchmark = True
    config = yaml_util.load_yaml_file(args.config)
    device = torch.device(args.device)
    dataset_dict = dataset_util.get_all_dataset(config['datasets'], args.use_cache)
    models_config = config['models']
    teacher_model_config = models_config['teacher_model']
    teacher_model = get_model(teacher_model_config, device, distributed, False)
    module_util.freeze_module_params(teacher_model)
    student_model_config = models_config['student_model']
    student_model = get_model(student_model_config, device, distributed, args.sync_bn)
    if distributed:
        teacher_model = DataParallel(teacher_model, device_ids=device_ids)
        student_model = DistributedDataParallel(student_model, device_ids=device_ids)

    start_epoch = args.start_epoch
    if not args.test_only:
        distill(teacher_model, student_model, dataset_dict, device, distributed, start_epoch, config, args)
        student_model_without_ddp =\
            student_model.module if isinstance(student_model, DistributedDataParallel) else student_model
        load_ckpt(student_model_config['ckpt'], model=student_model_without_ddp, strict=True)

    test_config = config['test']
    test_data_loader_config = test_config['test_data_loader']
    test_data_loader = dataset_util.build_data_loader(dataset_dict[test_data_loader_config['dataset_id']],
                                                      test_data_loader_config, distributed)
    if not args.student_only:
        evaluate(teacher_model, test_data_loader, device, title='[Teacher: {}]'.format(teacher_model_config['name']))
    evaluate(student_model, test_data_loader, device, title='[Student: {}]'.format(student_model_config['name']))


if __name__ == '__main__':
    argparser = get_argparser()
    main(argparser.parse_args())
