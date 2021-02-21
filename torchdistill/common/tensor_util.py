from collections import namedtuple

QuantizedTensor = namedtuple('QuantizedTensor', ['tensor', 'scale', 'zero_point'])


# Referred to https://github.com/eladhoffer/utils.pytorch/blob/master/quantize.py
#  and http://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf
def quantize_tensor(x, num_bits=8):
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
    return q_x.scale * (q_x.tensor.float() - q_x.zero_point)
