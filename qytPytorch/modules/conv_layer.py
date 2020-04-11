"""
    module(conv_layer) - 卷积常用层.

    Main members:

        # GlobalAvgPool2d - 全局平均池化层.
        # GlobalMaxPool1d - 时序最大池化.
"""
from torch import nn
import torch.nn.functional as F


class GlobalAvgPool2d(nn.Module):
    """全局平均池化层.可通过将池化窗口形状设置成输入的高和宽实现"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])


class GlobalMaxPool1d(nn.Module):
    """时序最大池化（max-over-time pooling）,对应一维全局最大池化层"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x shape: (batch_size, channel, seq_len)
        # return shape: (batch_size, channel, 1)
        return F.max_pool1d(x, kernel_size=x.shape[2])