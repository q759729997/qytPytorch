"""
    module(layer) - 常用层.

    Main members:

        # FlattenLayer - tensor展平层.
        # Inception - GoogLeNet中的基础卷积块Inception块.
        # Residual - 残差块.
        # DenseBlock - 稠密块.
"""
import torch
from torch import nn
import torch.nn.functional as F


class FlattenLayer(nn.Module):
    """tensor展平层,返回的shape为:(batch,展平后的数值)"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)


class Inception(nn.Module):
    """GoogLeNet中的基础卷积块Inception块"""
    # c1 - c4为每条线路里的层的输出通道数
    def __init__(self, in_c, c1, c2, c3, c4):
        super().__init__()
        # 线路1，单1 x 1卷积层
        self.p1_1 = nn.Conv2d(in_c, c1, kernel_size=1)
        # 线路2，1 x 1卷积层后接3 x 3卷积层
        self.p2_1 = nn.Conv2d(in_c, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3，1 x 1卷积层后接5 x 5卷积层
        self.p3_1 = nn.Conv2d(in_c, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4，3 x 3最大池化层后接1 x 1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_c, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1, p2, p3, p4), dim=1)  # 在通道维上连结输出


class Residual(nn.Module):
    """残差块"""
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)


class DenseBlock(nn.Module):
    """稠密块"""
    def __init__(self, num_convs, in_channels, out_channels):
        super().__init__()
        net = []
        for i in range(num_convs):
            in_c = in_channels + i * out_channels
            net.append(self._conv_block(in_c, out_channels))
        self.net = nn.ModuleList(net)
        self.out_channels = in_channels + num_convs * out_channels  # 计算输出通道数

    def _conv_block(self, in_channels, out_channels):
        blk = nn.Sequential(nn.BatchNorm2d(in_channels), 
                            nn.ReLU(),
                            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        return blk

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1)  # 在通道维上将输入和输出连结
        return X
