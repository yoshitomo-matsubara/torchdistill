from .registry import register_func2extract_org_output


@register_func2extract_org_output
def extract_simple_org_loss(org_criterion, student_outputs, targets, **kwargs):
    org_loss_dict = dict()
    if org_criterion is not None:
        # Models with auxiliary classifier returns multiple outputs
        if isinstance(student_outputs, (list, tuple)):
            for i, sub_outputs in enumerate(student_outputs):
                org_loss_dict[i] = org_criterion(sub_outputs, targets)
        else:
            org_loss = org_criterion(student_outputs, targets)
            org_loss_dict = {0: org_loss}
    return org_loss_dict


@register_func2extract_org_output
def extract_simple_org_loss_dict(org_criterion, student_outputs, targets, **kwargs):
    org_loss_dict = dict()
    if isinstance(student_outputs, dict):
        org_loss_dict = dict()
        for key, outputs in student_outputs.items():
            org_loss_dict[key] = org_criterion(outputs, targets)
    return org_loss_dict


@register_func2extract_org_output
def extract_org_loss_dict(org_criterion, student_outputs, targets, **kwargs):
    org_loss_dict = dict()
    if isinstance(student_outputs, dict):
        org_loss_dict.update(student_outputs)
    return org_loss_dict
