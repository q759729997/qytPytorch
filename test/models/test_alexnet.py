"""
    main_module - AlexNet，测试时将对应方法的@unittest.skip注释掉.

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
from qytPytorch.models.alexnet import AlexNet  # noqa
from qytPytorch.utils.statistics_utils import get_parameter_number  # noqa


class TestAlexNet(unittest.TestCase):
    """AlexNet.

    Main methods:
        test_get_conv_out_shape - 计算AlexNet卷积层输出形状.
        test_get_parameter_number - 统计神经网络参数个数.
    """
    @unittest.skip('debug')
    def test_get_conv_out_shape(self):
        """计算AlexNet卷积层输出形状.
        """
        print('{} test_get_Conv2d_out_shape {}'.format('-'*15, '-'*15))
        input_shape = (64, 1, 224, 224)  # 批量大小, 通道, 高, 宽
        conv_shapes = [
            ('conv', (1, 96, 11, 4)),  # in_channels, out_channels, kernel_size, stride, padding
            ('maxpool', (3, 2)),  # kernel_size, stride
            ('conv', (96, 256, 5, 1, 2)),
            ('maxpool', (3, 2)),
            ('conv', (256, 384, 3, 1, 1)),
            ('conv', (2384, 384, 3, 1, 1)),
            ('conv', (2384, 256, 3, 1, 1)),
            ('maxpool', (3, 2)),
        ]
        print('input_shape:{}'.format(input_shape))
        output_shape = input_shape
        for index, (conv_type, conv_shape) in enumerate(conv_shapes):
            if conv_type == 'conv':
                if len(conv_shape) == 4:
                    output_shape = get_Conv2d_out_shape(input_shape=output_shape, out_channels=conv_shape[1], kernel_size=conv_shape[2], stride=conv_shape[3])
                elif len(conv_shape) == 5:
                    output_shape = get_Conv2d_out_shape(input_shape=output_shape, out_channels=conv_shape[1], kernel_size=conv_shape[2], stride=conv_shape[3], padding=conv_shape[4])
            elif conv_type == 'maxpool':
                output_shape = get_MaxPool2d_out_shape(input_shape=output_shape, kernel_size=conv_shape[0], stride=conv_shape[1])
            print('layer:{} {}:{}, output_shape:{}'.format(index+1, conv_type, conv_shape, output_shape))
        print('output_shape:{}'.format(output_shape))
        """
        input_shape:(64, 1, 224, 224)
        layer:1 conv:(1, 96, 11, 4), output_shape:(64, 96, 54, 54)
        layer:2 maxpool:(3, 2), output_shape:(64, 96, 26, 26)
        layer:3 conv:(96, 256, 5, 1, 2), output_shape:(64, 256, 26, 26)
        layer:4 maxpool:(3, 2), output_shape:(64, 256, 12, 12)
        layer:5 conv:(256, 384, 3, 1, 1), output_shape:(64, 384, 12, 12)
        layer:6 conv:(2384, 384, 3, 1, 1), output_shape:(64, 384, 12, 12)
        layer:7 conv:(2384, 256, 3, 1, 1), output_shape:(64, 256, 12, 12)
        layer:8 maxpool:(3, 2), output_shape:(64, 256, 5, 5)
        output_shape:(64, 256, 5, 5)
        """
        x = torch.ones(input_shape)
        y = AlexNet().conv(x)
        print(y.shape)  # torch.Size([64, 256, 5, 5])

    # @unittest.skip('debug')
    def test_get_parameter_number(self):
        """ 统计神经网络参数个数.
        """
        print('{} test_get_parameter_number {}'.format('-'*15, '-'*15))
        net = AlexNet()
        print(net)
        print(get_parameter_number(net))  # {'total': 46764746, 'trainable': 46764746}


if __name__ == "__main__":
    unittest.main()  # 运行当前源文件中的所有测试用例
