"""
    main_module - CNN工具类，测试时将对应方法的@unittest.skip注释掉.

    Main members:

        # __main__ - 程序入口.
"""
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, './')  # 定义搜索路径的优先顺序，序号从0开始，表示最大优先级

import qytPytorch  # noqa
print('qytPytorch module path :{}'.format(qytPytorch.__file__))  # 输出测试模块文件位置
from qytPytorch.utils.cnn_utils import get_Conv2d_out_shape  # noqa
from qytPytorch.utils.cnn_utils import get_MaxPool2d_out_shape  # noqa


class TestCNN(unittest.TestCase):
    """CNN工具类.

    Main methods:
        test_get_Conv2d_out_shape - 计算2维卷积输出形状.
        test_get_MaxPool2d_out_shape - 计算2维池化层输出形状.
    """
    @unittest.skip('debug')
    def test_get_Conv2d_out_shape(self):
        """计算2维卷积输出形状.
        """
        print('{} test_get_Conv2d_out_shape {}'.format('-'*15, '-'*15))
        net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3)
        )
        x = torch.ones(8, 1, 28, 28)  # 批量大小, 通道, 高, 宽
        y = net(x)
        print(y.shape)  # torch.Size([8, 2, 26, 26])
        print(get_Conv2d_out_shape(input_shape=x.shape, out_channels=2, kernel_size=3))  # torch.Size([8, 2, 26, 26])
        net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=3, padding=1)
        )
        x = torch.ones(8, 1, 28, 28)  # 批量大小, 通道, 高, 宽
        y = net(x)
        print(y.shape)  # torch.Size([8, 2, 10, 10])
        print(get_Conv2d_out_shape(input_shape=x.shape, out_channels=2, kernel_size=3, stride=3, padding=1))  # (8, 2, 10, 10)

    # @unittest.skip('debug')
    def test_get_MaxPool2d_out_shape(self):
        """计算2维池化层输出形状.
        """
        print('{} test_get_MaxPool2d_out_shape {}'.format('-'*15, '-'*15))
        net = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1)  # the stride of the window. Default value is kernel_size
        )
        x = torch.ones(8, 1, 28, 28)  # 批量大小, 通道, 高, 宽
        y = net(x)
        print(y.shape)  # torch.Size([8, 1, 26, 26])
        print(get_MaxPool2d_out_shape(input_shape=x.shape, kernel_size=3))  # (8, 1, 26, 26)
        net = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=3, padding=1)
        )
        x = torch.ones(8, 1, 28, 28)  # 批量大小, 通道, 高, 宽
        y = net(x)
        print(y.shape)  # torch.Size([8, 1, 10, 10])
        print(get_MaxPool2d_out_shape(input_shape=x.shape, kernel_size=3, stride=3, padding=1))  # (8, 1, 10, 10)


if __name__ == "__main__":
    unittest.main()  # 运行当前源文件中的所有测试用例
