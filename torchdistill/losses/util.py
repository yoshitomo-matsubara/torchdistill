from .registry import register_func2extract_model_output


@register_func2extract_model_output
def extract_model_loss_dict(student_outputs, targets, **kwargs):
    model_loss_dict = dict()
    if isinstance(student_outputs, dict):
        model_loss_dict.update(student_outputs)
    return model_loss_dict
