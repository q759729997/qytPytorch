"""
    module(layer) - 常用层.

    Main members:

        # FlattenLayer - tensor展平层.
        # GlobalAvgPool2d - 全局平均池化层.
"""
from torch import nn
import torch.nn.functional as F


class FlattenLayer(nn.Module):
    """tensor展平层,返回的shape为:(batch,展平后的数值)"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)


class GlobalAvgPool2d(nn.Module):
    """全局平均池化层.可通过将池化窗口形状设置成输入的高和宽实现"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])

