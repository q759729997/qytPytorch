"""
    main_module - 损失函数绘制图像显示，测试是将对应方法的@unittest.skip注释掉.

    Main members:

        # __main__ - 程序入口.
"""
import unittest
import sys

import numpy as np
from matplotlib import pyplot as plt

sys.path.insert(0, './')  # 定义搜索路径的优先顺序，序号从0开始，表示最大优先级

import qytPytorch  # noqa
print('qytPytorch module path :{}'.format(qytPytorch.__file__))  # 输出测试模块文件位置
from qytPytorch.utils.matplotlib_utils import show_x_y_axis  # noqa


class TestLossDisplay(unittest.TestCase):
    """ 损失函数绘制图像显示.

    Main methods:
        test_local_minimum - 局部最小值显示.
        test_saddle_point - 鞍点.
        test_saddle_point_3d - 鞍点3D.
    """

    @unittest.skip('debug')
    def test_local_minimum(self):
        """ 局部最小值显示.
        """
        print('{} test_local_minimum {}'.format('-'*15, '-'*15))
        data_x = np.arange(-1.0, 2.0, 0.1)
        data_y = data_x * np.cos(np.pi * data_x)
        annotates = [
            {'text': 'local minimum', 'xy': (-0.3, -0.25), 'xytext': (-0.7, -1.0), 'arrowstyle': '->'},
            {'text': 'global minimum', 'xy': (1.1, -0.95), 'xytext': (0.6, 0.8), 'arrowstyle': '->'}
        ]
        show_x_y_axis(data_x, data_y, title='local minimum', annotates=annotates)  # 直接弹出图片

    @unittest.skip('debug')
    def test_saddle_point(self):
        """ 鞍点.
        """
        print('{} test_saddle_point {}'.format('-'*15, '-'*15))
        data_x = np.arange(-2.0, 2.0, 0.1)
        data_y = data_x ** 3
        annotates = [
            {'text': 'saddle point', 'xy': (0, -0.2), 'xytext': (-0.52, -5.0), 'arrowstyle': '->'}
        ]
        show_x_y_axis(data_x, data_y, title='saddle point', annotates=annotates)  # 直接弹出图片

    # @unittest.skip('debug')
    def test_saddle_point_3d(self):
        """ 鞍点3D.
        """
        print('{} test_saddle_point_3d {}'.format('-'*15, '-'*15))
        x, y = np.mgrid[-1:1: 31j, -1: 1: 31j]
        z = x**2 - y ** 2
        ax = plt.figure().add_subplot(111, projection='3d')
        ax.plot_wireframe(x, y, z, **{'rstride': 2, 'cstride': 2})
        ax.plot([0], [0], [0], 'rx')
        ticks = [-1,  0, 1]
        plt.xticks(ticks)
        plt.yticks(ticks)
        ax.set_zticks(ticks)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()


if __name__ == "__main__":
    unittest.main()  # 运行当前源文件中的所有测试用例
