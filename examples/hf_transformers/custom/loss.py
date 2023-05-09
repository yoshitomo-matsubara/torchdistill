from torchdistill.losses.util import register_func2extract_model_output


@register_func2extract_model_output
def extract_transformers_loss(student_outputs, targets, **kwargs):
    model_loss_dict = dict()
    model_loss_dict['loss'] = student_outputs.loss
    return model_loss_dict
