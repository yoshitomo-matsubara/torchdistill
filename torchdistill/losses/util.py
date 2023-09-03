from .registry import register_func2extract_model_output


@register_func2extract_model_output
def extract_model_loss_dict(student_outputs, targets, **kwargs):
    """
    Extracts model's loss dict.

    :param student_outputs: student model's output.
    :type student_outputs: Amy
    :param targets: training targets (won't be used).
    :type targets: Amy
    :return: registered function to extract model output.
    :rtype: dict
    """
    model_loss_dict = dict()
    if isinstance(student_outputs, dict):
        model_loss_dict.update(student_outputs)
    return model_loss_dict
