import argparse
import datetime
import os
import time

import torch
from torch import distributed as dist
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data._utils.collate import default_collate
from torchvision.models.detection.keypoint_rcnn import KeypointRCNN
from torchvision.models.detection.mask_rcnn import MaskRCNN

from kdkit.common.constant import def_logger
from kdkit.common.main_util import is_main_process, init_distributed_mode, load_ckpt, save_ckpt
from kdkit.datasets import util
from kdkit.datasets.coco import get_coco_api_from_dataset
from kdkit.eval.coco import CocoEvaluator
from kdkit.misc.log import setup_log_file, SmoothedValue, MetricLogger
from kdkit.models import MODEL_DICT
from kdkit.models.official import get_object_detection_model
from kdkit.tools.distillation import get_distillation_box
from myutils.common import file_util, yaml_util
from myutils.pytorch import module_util

logger = def_logger.getChild(__name__)


def get_argparser():
    parser = argparse.ArgumentParser(description='Knowledge distillation for object detection models')
    parser.add_argument('--config', required=True, help='yaml file path')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--log', help='log file path')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('-test_only', action='store_true', help='Only test the models')
    parser.add_argument('-student_only', action='store_true', help='Test the student model only')
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('-adjust_lr', action='store_true',
                        help='multiply learning rate by number of distributed processes (world_size)')
    return parser


def get_model(model_config, device):
    model = get_object_detection_model(model_config)
    if model is None:
        model = MODEL_DICT[model_config['name']](**model_config['params'])

    ckpt_file_path = model_config['ckpt']
    load_ckpt(ckpt_file_path, model=model, strict=True)
    return model.to(device)


def distill_one_epoch(distillation_box, device, epoch, log_freq):
    metric_logger = MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', SmoothedValue(window_size=10, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    for sample_batch, targets, supp_dict in \
            metric_logger.log_every(distillation_box.train_data_loader, log_freq, header):
        start_time = time.time()
        sample_batch = list(image.to(device) for image in sample_batch)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        supp_dict = default_collate(supp_dict)
        loss = distillation_box(sample_batch, targets, supp_dict)
        distillation_box.update_params(loss)
        batch_size = len(sample_batch)
        metric_logger.update(loss=loss.item(), lr=distillation_box.optimizer.param_groups[0]['lr'])
        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))


def get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, (DataParallel, DistributedDataParallel)):
        model_without_ddp = model.module

    iou_type_list = ['bbox']
    if isinstance(model_without_ddp, MaskRCNN):
        iou_type_list.append('segm')
    if isinstance(model_without_ddp, KeypointRCNN):
        iou_type_list.append('keypoints')
    return iou_type_list


@torch.no_grad()
def evaluate(model, data_loader, device, device_ids, distributed, log_freq=1000, title=None, header='Test:'):
    model.to(device)
    if distributed:
        model = DistributedDataParallel(model, device_ids=device_ids)
    elif device.type.startswith('cuda'):
        model = DataParallel(model, device_ids=device_ids)

    if title is not None:
        logger.info(title)

    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device('cpu')
    model.eval()
    metric_logger = MetricLogger(delimiter='  ')
    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)
    for sample_batch, targets in metric_logger.log_every(data_loader, log_freq, header):
        sample_batch = list(image.to(device) for image in sample_batch)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(sample_batch)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target['image_id'].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    avg_stats_str = 'Averaged stats: {}'.format(metric_logger)
    logger.info(avg_stats_str)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator


def distill(teacher_model, student_model, dataset_dict, device, device_ids, distributed, config, args):
    logger.info('Start distillation')
    train_config = config['train']
    lr_factor = args.world_size if distributed and args.adjust_lr else 1
    distillation_box =\
        get_distillation_box(teacher_model, student_model, dataset_dict,
                             train_config, device, device_ids, distributed, lr_factor)
    ckpt_file_path = config['models']['student_model']['ckpt']
    best_val_map = 0.0
    optimizer, lr_scheduler = distillation_box.optimizer, distillation_box.lr_scheduler
    if file_util.check_if_exists(ckpt_file_path):
        best_val_map, _, _ = load_ckpt(ckpt_file_path, optimizer=optimizer, lr_scheduler=lr_scheduler)

    log_freq = train_config['log_freq']
    student_model_without_ddp = student_model.module if module_util.check_if_wrapped(student_model) else student_model
    start_time = time.time()
    for epoch in range(args.start_epoch, distillation_box.num_epochs):
        distillation_box.pre_process(epoch=epoch)
        distill_one_epoch(distillation_box, device, epoch, log_freq)
        val_coco_evaluator =\
            evaluate(student_model, distillation_box.val_data_loader, device, device_ids, distributed,
                     log_freq=log_freq, header='Validation:')
        # Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]
        val_map = val_coco_evaluator.coco_eval['bbox'].stats[0]
        if val_map > best_val_map and is_main_process():
            logger.info('Updating ckpt (Best BBox mAP: {:.4f} -> {:.4f})'.format(best_val_map, val_map))
            best_val_map = val_map
            save_ckpt(student_model_without_ddp, optimizer, lr_scheduler,
                      best_val_map, config, args, ckpt_file_path)
        distillation_box.post_process()

    if distributed:
        dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))
    distillation_box.clean_modules()


def main(args):
    log_file_path = args.log
    if is_main_process() and log_file_path is not None:
        setup_log_file(os.path.expanduser(log_file_path))

    distributed, device_ids = init_distributed_mode(args.world_size, args.dist_url)
    logger.info(args)
    cudnn.benchmark = True
    config = yaml_util.load_yaml_file(os.path.expanduser(args.config))
    device = torch.device(args.device)
    dataset_dict = util.get_all_dataset(config['datasets'])
    models_config = config['models']
    teacher_model_config = models_config['teacher_model']
    teacher_model = get_model(teacher_model_config, device)
    student_model_config = models_config['student_model']
    student_model = get_model(student_model_config, device)
    if not args.test_only:
        distill(teacher_model, student_model, dataset_dict, device, device_ids, distributed, config, args)
        student_model_without_ddp =\
            student_model.module if module_util.check_if_wrapped(student_model) else student_model
        load_ckpt(student_model_config['ckpt'], model=student_model_without_ddp, strict=True)

    test_config = config['test']
    test_data_loader_config = test_config['test_data_loader']
    test_data_loader = util.build_data_loader(dataset_dict[test_data_loader_config['dataset_id']],
                                              test_data_loader_config, distributed)
    if not args.student_only:
        evaluate(teacher_model, test_data_loader, device, device_ids, distributed,
                 title='[Teacher: {}]'.format(teacher_model_config['name']))
    evaluate(student_model, test_data_loader, device, device_ids, distributed,
             title='[Student: {}]'.format(student_model_config['name']))


if __name__ == '__main__':
    argparser = get_argparser()
    main(argparser.parse_args())
