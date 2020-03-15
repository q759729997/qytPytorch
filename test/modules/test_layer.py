"""
    main_module - 常用层，测试是将对应方法的@unittest.skip注释掉.

    Main members:

        # __main__ - 程序入口.
"""
import unittest
import sys

import torch

sys.path.insert(0, './')  # 定义搜索路径的优先顺序，序号从0开始，表示最大优先级

import qytPytorch  # noqa
print('qytPytorch module path :{}'.format(qytPytorch.__file__))  # 输出测试模块文件位置
from qytPytorch.modules.layer import FlattenLayer  # noqa


class TestLayer(unittest.TestCase):
    """ 常用层.

    Main methods:
        test_FlattenLayer - tensor展平层.
    """

    # @unittest.skip('debug')
    def test_FlattenLayer(self):
        """ tensor展平层.
        """
        print('{} test_FlattenLayer {}'.format('-'*15, '-'*15))
        net = FlattenLayer()
        print(net)  # FlattenLayer()
        x = torch.ones(2, 3, 3)
        print(x.shape)  # torch.Size([2, 3, 3])
        y = net(x)
        print(y.shape)  # torch.Size([2, 9])


if __name__ == "__main__":
    unittest.main()  # 运行当前源文件中的所有测试用例
