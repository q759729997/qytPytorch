"""
    main_module - GoogLeNet，测试时将对应方法的@unittest.skip注释掉.

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
from qytPytorch.models.googlenet import GoogLeNet  # noqa
from qytPytorch.utils.statistics_utils import get_parameter_number  # noqa


class TestNiN(unittest.TestCase):
    """GoogLeNet.

    Main methods:
        test_get_conv_out_shape - 计算GoogLeNet卷积层输出形状.
        test_get_parameter_number - 统计神经网络参数个数.
    """
    # @unittest.skip('debug')
    def test_get_conv_out_shape(self):
        """计算GoogLeNet卷积层输出形状.
        """
        print('{} test_get_conv_out_shape {}'.format('-'*15, '-'*15))
        net = GoogLeNet()
        # print(net)
        input_shape = (1, 1, 224, 224)  # 批量大小, 通道, 高, 宽
        x = torch.ones(input_shape)
        y = net(x)
        print(y.shape)  # torch.Size([1, 10])

    # @unittest.skip('debug')
    def test_get_parameter_number(self):
        """ 统计神经网络参数个数.
        """
        print('{} test_get_parameter_number {}'.format('-'*15, '-'*15))
        net = GoogLeNet()
        print(net)
        print(get_parameter_number(net))  # {'total': 1992166, 'trainable': 1992166}


if __name__ == "__main__":
    unittest.main()  # 运行当前源文件中的所有测试用例
