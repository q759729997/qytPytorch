"""
    main_module - LeNet，测试时将对应方法的@unittest.skip注释掉.

    Main members:

        # __main__ - 程序入口.
"""
import sys
import unittest

import torch

sys.path.insert(0, './')  # 定义搜索路径的优先顺序，序号从0开始，表示最大优先级

import qytPytorch  # noqa
print('qytPytorch module path :{}'.format(qytPytorch.__file__))  # 输出测试模块文件位置
from qytPytorch.utils.cnn_utils import get_Conv2d_out_shape  # noqa
from qytPytorch.utils.cnn_utils import get_MaxPool2d_out_shape  # noqa
from qytPytorch.models.lenet import LeNet  # noqa
from qytPytorch.utils.statistics_utils import get_parameter_number  # noqa


class TestLeNet(unittest.TestCase):
    """LeNet.

    Main methods:
        test_get_conv_out_shape - 计算LeNet卷积层输出形状.
        test_get_parameter_number - 统计神经网络参数个数.
    """
    @unittest.skip('debug')
    def test_get_conv_out_shape(self):
        """计算LeNet卷积层输出形状.
        """
        print('{} test_get_Conv2d_out_shape {}'.format('-'*15, '-'*15))
        input_shape = (64, 1, 28, 28)  # 批量大小, 通道, 高, 宽
        conv_shapes = [
            ('conv', (1, 6, 5)),  # in_channels, out_channels, kernel_size
            ('maxpool', (2, 2)),  # kernel_size, stride
            ('conv', (6, 16, 5)),
            ('maxpool', (2, 2)),
        ]
        print('input_shape:{}'.format(input_shape))
        output_shape = input_shape
        for index, (conv_type, conv_shape) in enumerate(conv_shapes):
            if conv_type == 'conv':
                output_shape = get_Conv2d_out_shape(input_shape=output_shape, out_channels=conv_shape[1], kernel_size=conv_shape[2])
            elif conv_type == 'maxpool':
                output_shape = get_MaxPool2d_out_shape(input_shape=output_shape, kernel_size=conv_shape[0], stride=conv_shape[1])
            print('layer:{} {}:{}, output_shape:{}'.format(index+1, conv_type, conv_shape, output_shape))
        print('output_shape:{}'.format(output_shape))
        """
        input_shape:(64, 1, 28, 28)
        layer:1 conv:(1, 6, 5), output_shape:(64, 6, 24, 24)
        layer:2 maxpool:(2, 2), output_shape:(64, 6, 12, 12)
        layer:3 conv:(6, 16, 5), output_shape:(64, 16, 8, 8)
        layer:4 maxpool:(2, 2), output_shape:(64, 16, 4, 4)
        output_shape:(64, 16, 4, 4)
        """
        x = torch.ones(input_shape)
        y = LeNet().conv(x)
        print(y.shape)  # torch.Size([64, 16, 4, 4])

    # @unittest.skip('debug')
    def test_get_parameter_number(self):
        """ 统计神经网络参数个数.
        """
        print('{} test_get_parameter_number {}'.format('-'*15, '-'*15))
        net = LeNet()
        print(net)
        print(get_parameter_number(net))  # {'total': 44426, 'trainable': 44426}


if __name__ == "__main__":
    unittest.main()  # 运行当前源文件中的所有测试用例
