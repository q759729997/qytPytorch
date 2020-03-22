from torch import nn

from qytPytorch.modules.layer import GlobalAvgPool2d
from qytPytorch.modules.layer import FlattenLayer


class NiN(nn.Module):
    """串联多个由卷积层和“全连接”层构成的小网络来构建一个深层网络"""
    def __init__(self):
        super().__init__()
        # 输出形状：1, 1, 224, 224
        self.net = nn.Sequential(
            self._nin_block(1, 96, kernel_size=11, stride=4, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=2),
            self._nin_block(96, 256, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            self._nin_block(256, 384, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(0.5),
            # 标签类别数是10
            self._nin_block(384, 10, kernel_size=3, stride=1, padding=1),
            GlobalAvgPool2d(),
            # 将四维的输出转成二维的输出，其形状为(批量大小, 10)
            FlattenLayer())

    def _nin_block(self, in_channels, out_channels, kernel_size, stride, padding):
        """NiN块"""
        blk = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                            nn.ReLU(),
                            nn.Conv2d(out_channels, out_channels, kernel_size=1),
                            nn.ReLU(),
                            nn.Conv2d(out_channels, out_channels, kernel_size=1),
                            nn.ReLU())
        return blk

    def forward(self, img):
        output = self.net(img)
        return output
