from .registry import register_func2extract_org_output


@register_func2extract_org_output
def extract_simple_org_loss(org_criterion, student_outputs, teacher_outputs, targets, uses_teacher_output, **kwargs):
    org_loss_dict = dict()
    if org_criterion is not None:
        # Models with auxiliary classifier returns multiple outputs
        if isinstance(student_outputs, (list, tuple)):
            if uses_teacher_output:
                for i, sub_student_outputs, sub_teacher_outputs in enumerate(zip(student_outputs, teacher_outputs)):
                    org_loss_dict[i] = org_criterion(sub_student_outputs, sub_teacher_outputs, targets)
            else:
                for i, sub_outputs in enumerate(student_outputs):
                    org_loss_dict[i] = org_criterion(sub_outputs, targets)
        else:
            org_loss = org_criterion(student_outputs, teacher_outputs, targets) if uses_teacher_output \
                else org_criterion(student_outputs, targets)
            org_loss_dict = {0: org_loss}
    return org_loss_dict


@register_func2extract_org_output
def extract_simple_org_loss_dict(org_criterion, student_outputs, teacher_outputs, targets,
                                 uses_teacher_output, **kwargs):
    org_loss_dict = dict()
    if isinstance(student_outputs, dict):
        is_teacher_output_dict = isinstance(teacher_outputs, dict)
        org_loss_dict = dict()
        for key, outputs in student_outputs.items():
            if uses_teacher_output and is_teacher_output_dict and key in teacher_outputs:
                org_loss_dict[key] = org_criterion(outputs, teacher_outputs[key], targets)
            else:
                org_loss_dict[key] = org_criterion(outputs, targets)
    return org_loss_dict


@register_func2extract_org_output
def extract_org_loss_dict(org_criterion, student_outputs, teacher_outputs, targets, uses_teacher_output, **kwargs):
    org_loss_dict = dict()
    if isinstance(student_outputs, dict):
        org_loss_dict.update(student_outputs)
    return org_loss_dict
