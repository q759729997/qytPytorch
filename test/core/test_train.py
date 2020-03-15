"""
    main_module - 模型训练，测试是将对应方法的@unittest.skip注释掉.

    Main members:

        # __main__ - 程序入口.
"""
import unittest
import sys

import torch
from torch import nn

sys.path.insert(0, './')  # 定义搜索路径的优先顺序，序号从0开始，表示最大优先级

import qytPytorch  # noqa
print('qytPytorch module path :{}'.format(qytPytorch.__file__))  # 输出测试模块文件位置
from qytPytorch import logger  # noqa
from qytPytorch.core.train import train_net  # noqa
from qytPytorch.dataset.fashion_mnist import get_dataset  # noqa
from qytPytorch.dataset.fashion_mnist import get_labels_by_ids  # noqa
from qytPytorch.dataset.fashion_mnist import get_data_iter  # noqa
from qytPytorch.modules.layer import FlattenLayer  # noqa


class TestTrain(unittest.TestCase):
    """ 模型训练.

    Main methods:
        test_train_net - 训练神经网络.
    """

    # @unittest.skip('debug')
    def test_train_net(self):
        """ 训练神经网络.
        """
        print('{} test_train_net {}'.format('-'*15, '-'*15))
        data_path = './data/FashionMNIST'
        batch_size = 64
        num_inputs = 1 * 28 * 28
        num_outputs = 10
        max_epoch = 5
        logger.info('加载数据')
        mnist_train, mnist_test = get_dataset(data_path=data_path)
        train_iter, test_iter = get_data_iter(mnist_train, mnist_test, batch_size=batch_size)
        logger.info('定义网络')
        net = nn.Sequential()
        net.add_module('flatten', FlattenLayer())
        net.add_module('linear', nn.Linear(num_inputs, num_outputs))
        logger.info(net)
        logger.info('参数初始化')
        torch.nn.init.normal_(net.linear.weight, mean=0, std=0.01)
        torch.nn.init.constant_(net.linear.bias, val=0)
        logger.info('定义损失函数')
        loss_func = nn.CrossEntropyLoss()
        logger.info('定义优化器')
        # optimizer = torch.optim.SGD(net.parameters(), lr=0.03)
        optimizer = torch.optim.Adam(net.parameters())
        logger.info('模型训练')
        train_net(net, train_iter, test_iter, max_epoch, optimizer, loss_func)
        """
        2020-03-15 23:27:49 I [train.py:24] epoch 1 begin to train
        2020-03-15 23:28:01 I [train.py:37] epoch 1, loss 0.0101, train acc 0.788, test acc 0.813
        2020-03-15 23:28:01 I [train.py:24] epoch 2 begin to train
        2020-03-15 23:28:13 I [train.py:37] epoch 2, loss 0.0075, train acc 0.838, test acc 0.836
        2020-03-15 23:28:13 I [train.py:24] epoch 3 begin to train
        2020-03-15 23:28:25 I [train.py:37] epoch 3, loss 0.0070, train acc 0.847, test acc 0.833
        2020-03-15 23:28:25 I [train.py:24] epoch 4 begin to train
        2020-03-15 23:28:37 I [train.py:37] epoch 4, loss 0.0067, train acc 0.852, test acc 0.840
        2020-03-15 23:28:37 I [train.py:24] epoch 5 begin to train
        2020-03-15 23:28:49 I [train.py:37] epoch 5, loss 0.0066, train acc 0.856, test acc 0.839
        """


if __name__ == "__main__":
    unittest.main()  # 运行当前源文件中的所有测试用例
