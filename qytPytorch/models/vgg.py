from torch import nn


class VGG11(nn.Module):
    """VGG-11：通过重复使用简单的基础块来构建深度模型"""
    def __init__(self):
        super().__init__()
        conv_arch = ((1, 1, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512), (2, 512, 512))
        # ratio = 8
        # small_conv_arch = [(1, 1, 64//ratio), (1, 64//ratio, 128//ratio), (2, 128//ratio, 256//ratio), 
        #            (2, 256//ratio, 512//ratio), (2, 512//ratio, 512//ratio)]
        # 经过5个vgg_block, 宽高会减半5次, 变成 224/32 = 7
        fc_features = 512 * 7 * 7  # c * w * h
        fc_hidden_units = 4096  # 任意
        # 定义VGG网络
        # 卷积层部分
        self.conv = nn.Sequential()
        for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch):
            # 每经过一个vgg_block都会使宽高减半
            self.conv.add_module("vgg_block_" + str(i+1), self._vgg_block(num_convs, in_channels, out_channels))
        # 全连接层部分
        # 输入形状：1, 1, 224, 224
        self.fc = nn.Sequential(
            nn.Linear(fc_features, fc_hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(fc_hidden_units, fc_hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(fc_hidden_units, 10)
        )

    def _vgg_block(self, num_convs, in_channels, out_channels):
        """VGG块"""
        blk = []
        for i in range(num_convs):
            if i == 0:
                blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            else:
                blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
            blk.append(nn.ReLU())
        blk.append(nn.MaxPool2d(kernel_size=2, stride=2))  # 这里会使宽高减半
        return nn.Sequential(*blk)

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output
