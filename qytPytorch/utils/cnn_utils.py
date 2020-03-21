"""
    module(cnn_utils) - CNN工具类.

    Main members:

        # get_Conv2d_out_shape - 计算2维卷积输出形状.
        # get_MaxPool2d_out_shape - 计算2维池化层输出形状.
"""
import math


def get_Conv2d_out_shape(input_shape, out_channels=1, kernel_size=1, stride=1, padding=0):
    """ 计算2维卷积输出形状.

        @params:
            input_shape - (批量大小, 通道, 高, 宽).
            out_channels - 输出channel个数.
            kernel_size - 卷积核大小.
            stride - 步长.
            padding - 填充.

        @return:
            On success - (批量大小, 通道, 高, 宽).
            On failure - 错误信息.
    """
    height = math.floor((input_shape[2] - kernel_size + 2*padding + stride) / stride)
    width = math.floor((input_shape[3] - kernel_size + 2*padding + stride) / stride)
    output_shape = (input_shape[0], out_channels, height, width)
    return output_shape


def get_MaxPool2d_out_shape(input_shape, kernel_size=1, stride=1, padding=0):
    """ 计算2维池化层输出形状.

        @params:
            input_shape - (批量大小, 通道, 高, 宽).
            kernel_size - 卷积核大小.
            stride - 步长.
            padding - 填充.

        @return:
            On success - (批量大小, 通道, 高, 宽).
            On failure - 错误信息.
    """
    height = math.floor((input_shape[2] - kernel_size + 2*padding + stride) / stride)
    width = math.floor((input_shape[3] - kernel_size + 2*padding + stride) / stride)
    output_shape = (input_shape[0], input_shape[1], height, width)
    return output_shape
