"""
    main_module - 准确率评估，测试是将对应方法的@unittest.skip注释掉.

    Main members:

        # __main__ - 程序入口.
"""
import unittest
import sys

import torch

sys.path.insert(0, './')  # 定义搜索路径的优先顺序，序号从0开始，表示最大优先级

import qytPytorch  # noqa
print('qytPytorch module path :{}'.format(qytPytorch.__file__))  # 输出测试模块文件位置
from qytPytorch.metric.accuracy import get_accuracy_score  # noqa


class TestAccuracy(unittest.TestCase):
    """ 准确率评估.

    Main methods:
        test_get_accuracy_score - 计算准确率分数.
    """

    # @unittest.skip('debug')
    def test_get_accuracy_score(self):
        """ 计算准确率分数.
        """
        print('{} test_get_accuracy_score {}'.format('-'*15, '-'*15))
        y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
        y = torch.tensor([2, 2])
        print(get_accuracy_score(y_hat, y))  # 1.0
        y = torch.tensor([1, 2])
        print(get_accuracy_score(y_hat, y))  # 0.5


if __name__ == "__main__":
    unittest.main()  # 运行当前源文件中的所有测试用例
