from torch import nn
from torch.nn import DataParallel
from torch.nn.parallel.distributed import DistributedDataParallel

from myutils.pytorch import module_util
from tools.loss import KDLoss, get_single_loss, get_custom_loss


def set_distillation_box_info(info_dict, module_path, **kwargs):
    info_dict[module_path] = kwargs


def register_forward_hook_with_dict(module, module_path, info_dict):
    def forward_hook(self, input, output):
        info_dict[module_path]['output'] = output
    return module.register_forward_hook(forward_hook)


class DistillationBox(nn.Module):
    def setup(self, criterion_config):
        self.target_module_pairs.clear()
        self.target_module_handles.clear()
        sub_terms_config = criterion_config.get('sub_terms', None)
        if sub_terms_config is not None:
            # TODO: Build teacher and student models with DP/DDP here; Teacher doesn't require grad
            teacher_model = self.org_teacher_model
            student_model = self.org_student_model

            teacher_model_without_dp =\
                teacher_model.module if isinstance(teacher_model, DataParallel) else teacher_model
            student_model_without_ddp = \
                student_model.module if isinstance(student_model, DistributedDataParallel) else student_model
            for loss_name, loss_config in sub_terms_config.items():
                teacher_path, student_path = loss_config['ts_modules']
                self.target_module_pairs.append((teacher_path, student_path))
                teacher_module = module_util.get_module(teacher_model_without_dp, teacher_path)
                student_module = module_util.get_module(student_model_without_ddp, student_path)
                set_distillation_box_info(self.teacher_info_dict, teacher_path, loss_name=loss_name,
                                          path_from_root=teacher_path, is_teacher=True)
                set_distillation_box_info(self.student_info_dict, student_path, loss_name=loss_name,
                                          path_from_root=student_path, is_teacher=False)
                teacher_handle = register_forward_hook_with_dict(teacher_module, teacher_path, self.teacher_info_dict)
                student_handle = register_forward_hook_with_dict(student_module, student_path, self.student_info_dict)
                self.target_module_handles.append((teacher_handle, student_handle))

        org_term_config = criterion_config.get('org_term', dict())
        org_criterion_config = org_term_config.get('criterion', dict()) if isinstance(org_term_config, dict) else None
        if org_criterion_config is not None and len(org_criterion_config) > 0:
            self.org_criterion = get_single_loss(org_criterion_config)

        self.criterion = get_custom_loss(criterion_config)
        self.use_teacher_output = isinstance(self.org_criterion, KDLoss)

    def __init__(self, teacher_model, student_model, criterion_config):
        super().__init__()
        self.org_teacher_model = teacher_model
        self.org_student_model = student_model
        self.target_module_pairs, self.target_module_handles = list(), list()
        self.teacher_info_dict, self.student_info_dict = dict(), dict()
        self.org_criterion, self.criterion, self.use_teacher_output = None, None, None
        self.setup(criterion_config)

    def check_if_org_loss_required(self):
        return self.org_criterion is not None

    def forward(self, sample_batch, targets):
        teacher_outputs = self.teacher_model(sample_batch)
        student_outputs = self.student_model(sample_batch)
        org_loss_dict = dict()
        if self.check_if_org_loss_required():
            # Models with auxiliary classifier returns multiple outputs
            if isinstance(student_outputs, (list, tuple)):
                if self.use_teacher_output:
                    for i, sub_student_outputs, sub_teacher_outputs in enumerate(zip(student_outputs, teacher_outputs)):
                        org_loss_dict[i] = self.org_criterion(sub_student_outputs, sub_teacher_outputs, targets)
                else:
                    for i, sub_outputs in enumerate(student_outputs):
                        org_loss_dict[i] = self.org_criterion(sub_outputs, targets)
            else:
                org_loss = self.org_criterion(student_outputs, teacher_outputs, targets) if self.use_teacher_output\
                    else self.org_criterion(student_outputs, targets)
                org_loss_dict = {0: org_loss}

        output_dict = dict()
        for teacher_path, student_path in self.target_module_pairs:
            teacher_module_dict = self.teacher_info_dict[teacher_path]
            student_module_dict = self.student_info_dict[student_path]
            output_dict[teacher_module_dict['loss_name']] = (
                (teacher_module_dict['path_from_root'], teacher_module_dict.pop('output')),
                (student_module_dict['path_from_root'], student_module_dict.pop('output'))
            )

        total_loss = self.criterion(output_dict, org_loss_dict)
        return total_loss

    def post_process(self, **kwargs):
        pass

    def clean_modules(self):
        for teacher_handle, student_handle in self.target_module_handles:
            teacher_handle.remove()
            student_handle.remove()


class MultiStagesDistillationBox(DistillationBox):
    def __init__(self, teacher_model, student_model, criterion_config):
        stage1_config = criterion_config['stage1']
        super().__init__(teacher_model, student_model, stage1_config['criterion'])
        self.criterion_config = criterion_config
        self.stage_number = 1
        self.stage_end_epoch = stage1_config['end_epoch']
        print('Stage {}'.format(self.stage_number))

    def advance_to_next_stage(self):
        for teacher_handle, student_handle in self.target_module_handles:
            teacher_handle.remove()
            student_handle.remove()

        self.stage_number += 1
        next_stage_config = self.criterion_config['stage{}'.format(self.stage_number)]
        self.setup(next_stage_config['criterion'])
        self.stage_end_epoch = next_stage_config['end_epoch']
        print('Advanced to stage {}'.format(self.stage_number))

    def post_process(self, epoch, **kwargs):
        if epoch == self.stage_end_epoch:
            self.advance_to_next_stage()


def get_distillation_box(teacher_model, student_model, main_criterion_config):
    if 'stage1' in main_criterion_config:
        return MultiStagesDistillationBox(teacher_model, student_model, main_criterion_config)
    return DistillationBox(teacher_model, student_model, main_criterion_config)
