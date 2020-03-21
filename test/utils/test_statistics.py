"""
    main_module - 神经网络分析统计工具类，测试是将对应方法的@unittest.skip注释掉.

    Main members:

        # __main__ - 程序入口.
"""
import unittest
import sys

from torch import nn

sys.path.insert(0, './')  # 定义搜索路径的优先顺序，序号从0开始，表示最大优先级

import qytPytorch  # noqa
print('qytPytorch module path :{}'.format(qytPytorch.__file__))  # 输出测试模块文件位置
from qytPytorch.utils.statistics_utils import get_parameter_number  # noqa


class TestStatistics(unittest.TestCase):
    """ 神经网络分析统计工具类.

    Main methods:
        test_get_parameter_number - 统计神经网络参数个数.
    """

    # @unittest.skip('debug')
    def test_get_parameter_number(self):
        """ 统计神经网络参数个数.
        """
        print('{} test_get_parameter_number {}'.format('-'*15, '-'*15))
        net = nn.Sequential(
                            nn.Linear(2, 20),
                            nn.Conv2d(1, 20, 5),
                            nn.ReLU(),
                            nn.Conv2d(20, 64, 5),
                            nn.ReLU()
                            )
        net[0].requires_grad_(False)  # 冻结第一层
        print(net)
        print(get_parameter_number(net))  # {'total': 32644, 'trainable': 32584}


if __name__ == "__main__":
    unittest.main()  # 运行当前源文件中的所有测试用例
