from collections import namedtuple

QuantizedTensor = namedtuple('QuantizedTensor', ['tensor', 'scale', 'zero_point'])


# Referred to https://github.com/eladhoffer/utils.pytorch/blob/master/quantize.py
#  and http://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf
def quantize_tensor(x, num_bits=8):
    """
    Quantizes a tensor using `num_bits` int and float.

    Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, Dmitry Kalenichenko: `"Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" <https://openaccess.thecvf.com/content_cvpr_2018/html/Jacob_Quantization_and_Training_CVPR_2018_paper.html>`_ @ CVPR 2018 (2018)

    :param x: tensor to be quantized.
    :type x: torch.Tensor
    :param num_bits: the number of bits for quantization.
    :type num_bits: int
    :return: quantized tensor.
    :rtype: QuantizedTensor
    """
    qmin = 0.0
    qmax = 2.0 ** num_bits - 1.0
    min_val, max_val = x.min(), x.max()
    scale = (max_val - min_val) / (qmax - qmin)
    initial_zero_point = qmin - min_val / scale
    zero_point = qmin if initial_zero_point < qmin else qmax if initial_zero_point > qmax else initial_zero_point
    zero_point = int(zero_point)
    qx = zero_point + x / scale
    qx = qx.clamp(qmin, qmax).round().byte()
    return QuantizedTensor(tensor=qx, scale=scale, zero_point=zero_point)


def dequantize_tensor(q_x):
    """
    Dequantizes a quantized tensor.

    Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, Dmitry Kalenichenko: `"Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" <https://openaccess.thecvf.com/content_cvpr_2018/html/Jacob_Quantization_and_Training_CVPR_2018_paper.html>`_ @ CVPR 2018 (2018)

    :param q_x: quantized tensor to be dequantized.
    :type q_x: QuantizedTensor
    :return: dequantized tensor.
    :rtype: torch.Tensor
    """
    return q_x.scale * (q_x.tensor.float() - q_x.zero_point)
