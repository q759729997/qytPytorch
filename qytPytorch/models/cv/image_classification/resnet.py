from torch import nn

from qytPytorch.modules.layer import Residual
from qytPytorch.modules.layer import FlattenLayer
from qytPytorch.modules.layer import GlobalAvgPool2d


class ResNet18(nn.Module):
    """残差模型,卷积层和最后的全连接层，共计18层,通常也被称为ResNet-18"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        # 残差块,每个模块使用两个残差块
        self.net.add_module("resnet_block1", self._resnet_block(64, 64, 2, first_block=True))
        self.net.add_module("resnet_block2", self._resnet_block(64, 128, 2))
        self.net.add_module("resnet_block3", self._resnet_block(128, 256, 2))
        self.net.add_module("resnet_block4", self._resnet_block(256, 512, 2))
        # 全连接层
        self.net.add_module("global_avg_pool", GlobalAvgPool2d())  # GlobalAvgPool2d的输出: (Batch, 512, 1, 1)
        self.net.add_module("fc", nn.Sequential(FlattenLayer(), nn.Linear(512, 10)))

    def _resnet_block(self, in_channels, out_channels, num_residuals, first_block=False):
        """使用若干个同样输出通道数的残差块。第一个模块的通道数同输入通道数一致。由于之前已经使用了步幅为2的最大池化层，所以无须减小高和宽。之后的每个模块在第一个残差块里将上一个模块的通道数翻倍，并将高和宽减半。"""
        if first_block:
            assert in_channels == out_channels  # 第一个模块的通道数同输入通道数一致
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
            else:
                blk.append(Residual(out_channels, out_channels))
        return nn.Sequential(*blk)

    def forward(self, img):
        output = self.net(img)
        return output
