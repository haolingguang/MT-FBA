import torch
import random
import numpy as np


def bit_depth_reduce(xs, x_min, x_max, step_num, alpha=1e6):
    ''' Run bit depth reduce on xs.

    :param xs: A batch of images to apply bit depth reduction.
    :param x_min: The minimum value of xs.
    :param x_max: The maximum value of xs.
    :param step_num: Step number for bit depth reduction.
    :param alpha: Alpha for bit depth reduction.
    :return: Bit depth reduced xs.
    '''

    def bit_depth_reduce_op(xs_tf):
        steps = x_min + np.arange(1, step_num, dtype=np.float32) / (step_num / (x_max - x_min))
        steps = torch.from_numpy(steps)
        steps = steps.view([1, 1, 1, 1, step_num-1])
        tf_steps = steps.float().cuda()

        inputs = xs_tf.unsqueeze(4)
        quantized_inputs = x_min + torch.sum(torch.sigmoid(alpha * (inputs - tf_steps)), dim=4)
        quantized_inputs = quantized_inputs / ((step_num-1) / (x_max - x_min))

        def bit_depth_reduce_grad(d_output):
            return d_output

        return quantized_inputs, bit_depth_reduce_grad

    return bit_depth_reduce_op(xs)

