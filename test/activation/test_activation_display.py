"""
    main_module - 激活函数绘制图像显示，测试是将对应方法的@unittest.skip注释掉.

    Main members:

        # __main__ - 程序入口.
"""
import unittest
import sys

import torch

sys.path.insert(0, './')  # 定义搜索路径的优先顺序，序号从0开始，表示最大优先级

import qytPytorch  # noqa
print('qytPytorch module path :{}'.format(qytPytorch.__file__))  # 输出测试模块文件位置
from qytPytorch.utils.matplotlib_utils import show_x_y_axis  # noqa


class TestActivationDisplay(unittest.TestCase):
    """ 激活函数绘制图像显示.

    Main methods:
        test_ReLU - ReLU函数显示.
        test_ReLU_grad - ReLU函数梯度显示.
    """

    @unittest.skip('debug')
    def test_ReLU(self):
        """ ReLU函数显示.
        """
        print('{} test_ReLU {}'.format('-'*15, '-'*15))
        data_x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
        data_y = data_x.relu()
        show_x_y_axis(data_x.detach().numpy(), data_y.detach().numpy(), title='ReLU')  # 直接弹出图片

    @unittest.skip('debug')
    def test_ReLU_grad(self):
        """ ReLU函数梯度显示.
        """
        print('{} test_ReLU {}'.format('-'*15, '-'*15))
        data_x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
        data_x.relu().sum().backward()
        data_y = data_x.grad
        show_x_y_axis(data_x.detach().numpy(), data_y.detach().numpy(), title='ReLU grad')  # 直接弹出图片


if __name__ == "__main__":
    unittest.main()  # 运行当前源文件中的所有测试用例
