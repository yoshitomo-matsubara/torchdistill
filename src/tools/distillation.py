from torch import nn

from myutils.pytorch import module_util
from tools.loss import KDLoss, get_single_loss, get_custom_loss


class DistillationBox(nn.Module):
    def __init__(self, teacher_model, student_model, criterion_config):
        super().__init__()
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.target_module_pairs = list()

        def extract_output(self, input, output):
            self.__dict__['distillation_box']['output'] = output

        for loss_name, loss_config in criterion_config['sub_terms'].items():
            teacher_path, student_path = loss_config['ts_modules']
            self.target_module_pairs.append((teacher_path, student_path))
            teacher_module = module_util.get_module(self.teacher_model, teacher_path)
            student_module = module_util.get_module(self.student_model, student_path)
            teacher_module.__dict__['distillation_box'] = {'loss_name': loss_name, 'path_from_root': teacher_path,
                                                           'is_teacher': True}
            student_module.__dict__['distillation_box'] = {'loss_name': loss_name, 'path_from_root': student_path,
                                                           'is_teacher': False}
            teacher_module.register_forward_hook(extract_output)
            student_module.register_forward_hook(extract_output)

        org_term_config = criterion_config['org_term']
        org_criterion_config = org_term_config['criterion']
        self.org_criterion = get_single_loss(org_criterion_config)
        self.org_factor = org_term_config['factor']
        self.criterion = get_custom_loss(criterion_config)

    def forward(self, sample_batch, targets):
        teacher_outputs = self.teacher_model(sample_batch)
        student_outputs = self.student_model(sample_batch)
        # Model with auxiliary classifier returns multiple outputs
        if isinstance(student_outputs, (list, tuple)):
            org_loss_dict = dict()
            if isinstance(self.org_criterion, KDLoss):
                for i, sub_student_outputs, sub_teacher_outputs in enumerate(zip(student_outputs, teacher_outputs)):
                    org_loss_dict[i] = self.org_criterion(sub_student_outputs, sub_teacher_outputs, targets)
            else:
                for i, sub_outputs in enumerate(student_outputs):
                    org_loss_dict[i] = self.org_criterion(sub_outputs, targets)
        else:
            org_loss_dict = {0: self.org_criterion(student_outputs, targets)}

        output_dict = dict()
        for teacher_path, student_path in self.target_module_pairs:
            teacher_dict = module_util.get_module(self.teacher_model, teacher_path).__dict__['distillation_box']
            student_dict = module_util.get_module(self.student_model, student_path).__dict__['distillation_box']
            output_dict[teacher_dict['loss_name']] = ((teacher_dict['path_from_root'], teacher_dict['output']),
                                                      (student_dict['path_from_root'], student_dict['output']))

        total_loss = self.criterion(output_dict, org_loss_dict)
        return total_loss
