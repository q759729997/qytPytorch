from torch import nn

from qytPytorch.modules.layer import DenseBlock
from qytPytorch.modules.layer import FlattenLayer
from qytPytorch.modules.layer import GlobalAvgPool2d


class DenseNet(nn.Module):
    """稠密网络模型"""
    def __init__(self):
        super().__init__()
        # 首先使用同ResNet一样的单卷积层和最大池化层
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        # 类似于ResNet接下来使用的4个残差块，DenseNet使用的是4个稠密块
        # 稠密块里的卷积层通道数（即增长率）设为32，所以每个稠密块将增加128个通道
        num_channels, growth_rate = 64, 32  # num_channels为当前的通道数
        num_convs_in_dense_blocks = [4, 4, 4, 4]
        for i, num_convs in enumerate(num_convs_in_dense_blocks):
            DB = DenseBlock(num_convs, num_channels, growth_rate)
            self.net.add_module("DenseBlosk_%d" % i, DB)
            # 上一个稠密块的输出通道数
            num_channels = DB.out_channels
            # 在稠密块之间加入通道数减半的过渡层
            if i != len(num_convs_in_dense_blocks) - 1:
                self.net.add_module("transition_block_%d" % i, self._transition_block(num_channels, num_channels // 2))
                num_channels = num_channels // 2
        # 最后接上全局池化层和全连接层来输出
        self.net.add_module("BN", nn.BatchNorm2d(num_channels))
        self.net.add_module("relu", nn.ReLU())
        self.net.add_module("global_avg_pool", GlobalAvgPool2d())  # GlobalAvgPool2d的输出: (Batch, num_channels, 1, 1)
        self.net.add_module("fc", nn.Sequential(FlattenLayer(), nn.Linear(num_channels, 10)))

    def _transition_block(self, in_channels, out_channels):
        """过渡层,过渡层用来控制模型复杂度。它通过1×1卷积层来减小通道数，并使用步幅为2的平均池化层减半高和宽，从而进一步降低模型复杂度。"""
        blk = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.AvgPool2d(kernel_size=2, stride=2))
        return blk

    def forward(self, img):
        output = self.net(img)
        return output
