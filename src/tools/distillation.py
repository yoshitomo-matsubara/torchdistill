from torch import nn
from torch.nn import DataParallel
from torch.nn.parallel.distributed import DistributedDataParallel

from myutils.pytorch import module_util
from tools.loss import KDLoss, get_single_loss, get_custom_loss


def set_distillation_box_info(module, **kwargs):
    module.__dict__['distillation_box'] = kwargs


def get_distillation_box_info(module):
    return module.__dict__['distillation_box']


def extract_output(self, input, output):
    self.__dict__['distillation_box']['output'] = output


class DistillationBox(nn.Module):
    def __init__(self, teacher_model, student_model, criterion_config):
        super().__init__()
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.target_module_pairs = list()
        sub_terms_config = criterion_config.get('sub_terms', None)
        if sub_terms_config is not None:
            teacher_model_without_dp =\
                teacher_model.module if isinstance(teacher_model, DataParallel) else teacher_model
            student_model_without_ddp = \
                student_model.module if isinstance(student_model, DistributedDataParallel) else student_model
            for loss_name, loss_config in sub_terms_config.items():
                teacher_path, student_path = loss_config['ts_modules']
                self.target_module_pairs.append((teacher_path, student_path))
                teacher_module = module_util.get_module(teacher_model_without_dp, teacher_path)
                student_module = module_util.get_module(student_model_without_ddp, student_path)
                set_distillation_box_info(teacher_module, loss_name=loss_name, path_from_root=teacher_path,
                                          is_teacher=True)
                set_distillation_box_info(student_module, loss_name=loss_name, path_from_root=student_path,
                                          is_teacher=False)
                teacher_module.register_forward_hook(extract_output)
                student_module.register_forward_hook(extract_output)

        org_term_config = criterion_config['org_term']
        org_criterion_config = org_term_config['criterion']
        self.org_criterion = get_single_loss(org_criterion_config)
        self.org_factor = org_term_config['factor']
        self.criterion = get_custom_loss(criterion_config)
        self.use_teacher_output = isinstance(self.org_criterion, KDLoss)

    def forward(self, sample_batch, targets):
        teacher_outputs = self.teacher_model(sample_batch)
        student_outputs = self.student_model(sample_batch)
        # Model with auxiliary classifier returns multiple outputs
        if isinstance(student_outputs, (list, tuple)):
            org_loss_dict = dict()
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
        teacher_model_without_dp = \
            self.teacher_model.module if isinstance(self.teacher_model, DataParallel) else self.teacher_model
        student_model_without_ddp = \
            self.student_model.module if isinstance(self.student_model, DistributedDataParallel) else self.student_model
        for teacher_path, student_path in self.target_module_pairs:
            teacher_dict = get_distillation_box_info(module_util.get_module(teacher_model_without_dp, teacher_path))
            student_dict = get_distillation_box_info(module_util.get_module(student_model_without_ddp, student_path))
            output_dict[teacher_dict['loss_name']] = ((teacher_dict['path_from_root'], teacher_dict.pop('output')),
                                                      (student_dict['path_from_root'], student_dict.pop('output')))

        total_loss = self.criterion(output_dict, org_loss_dict)
        return total_loss
