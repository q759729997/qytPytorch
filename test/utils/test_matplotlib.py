"""
    main_module - matplotlib常用工具函数，测试是将对应方法的@unittest.skip注释掉.

    Main members:

        # __main__ - 程序入口.
"""
import unittest
import sys

import numpy as np

sys.path.insert(0, './')  # 定义搜索路径的优先顺序，序号从0开始，表示最大优先级

import qytPytorch  # noqa
print('qytPytorch module path :{}'.format(qytPytorch.__file__))  # 输出测试模块文件位置
from qytPytorch.utils.matplotlib_utils import show_x_y_axis  # noqa


class TestMatplotlib(unittest.TestCase):
    """ matplotlib常用工具函数.

    Main methods:
        test_show_x_y_axis - 显示x、y坐标系.
    """

    # @unittest.skip('debug')
    def test_show_x_y_axis(self):
        """ 显示x、y坐标系.
        """
        print('{} test_show_x_y_axis {}'.format('-'*15, '-'*15))
        data_x = np.arange(0, 10)
        data_y = 2 * data_x + 5
        show_x_y_axis(data_x, data_y)  # 直接弹出图片


if __name__ == "__main__":
    unittest.main()  # 运行当前源文件中的所有测试用例
