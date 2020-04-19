"""
    main_module - CIFAR-10数据集常用函数，测试是将对应方法的@unittest.skip注释掉.

    Main members:

        # __main__ - 程序入口.
"""
import unittest
import sys
import time

sys.path.insert(0, './')  # 定义搜索路径的优先顺序，序号从0开始，表示最大优先级

import qytPytorch  # noqa
print('qytPytorch module path :{}'.format(qytPytorch.__file__))  # 输出测试模块文件位置
from qytPytorch.dataset.cv.image_classification.cifar10 import get_dataset  # noqa
from qytPytorch.dataset.cv.image_classification.cifar10 import get_labels_by_ids  # noqa
from qytPytorch.dataset.cv.image_classification.cifar10 import show_fashion_cifar10  # noqa
from qytPytorch.dataset.cv.image_classification.cifar10 import get_data_iter  # noqa


class TestCIFAR10(unittest.TestCase):
    """ CIFAR10数据集常用函数.

    Main methods:
        test_get_dataset - 获取数据集.
        test_get_labels_by_ids - 根据标签id获取标签具体描述.
        test_show_fashion_cifar10 - 展示图像与标签.
        test_get_data_iter - 获取数据集迭代器.
    """
    data_path = './data/cv/CIFAR10'

    @unittest.skip('debug')
    def test_get_dataset(self):
        """ 获取数据集.
        """
        print('{} test_get_dataset {}'.format('-'*15, '-'*15))
        mnist_train, mnist_test = get_dataset(data_path=self.data_path)
        feature, label = mnist_train[0]
        print(feature.shape, label)  # torch.Size([3, 32, 32]) 6

    @unittest.skip('debug')
    def test_get_labels_by_ids(self):
        """ 根据标签id获取标签具体描述.
        """
        print('{} test_get_labels_by_ids {}'.format('-'*15, '-'*15))
        label_ids = [1, 5, 3]
        print(get_labels_by_ids(label_ids))  # ['automobile', 'dog', 'cat']
        print(get_labels_by_ids(label_ids, return_Chinese=True))  # ['汽车', '狗', '猫']

    # @unittest.skip('debug')
    def test_show_fashion_cifar10(self):
        """ 展示图像与标签.
        """
        print('{} test_show_fashion_cifar10 {}'.format('-'*15, '-'*15))
        cifar10_train, cifar10_test = get_dataset(data_path=self.data_path)
        feature, label = cifar10_train[0]
        X, y = [], []
        for i in range(10):
            X.append(cifar10_train[i][0])
            y.append(cifar10_train[i][1])
        show_fashion_cifar10(X, get_labels_by_ids(y))   # 直接弹出图片展示页面

    @unittest.skip('debug')
    def test_get_data_iter(self):
        """ 获取数据集迭代器.
        """
        print('{} test_get_data_iter {}'.format('-'*15, '-'*15))
        mnist_train, mnist_test = get_dataset(data_path=self.data_path)
        train_iter, test_iter = get_data_iter(mnist_train, mnist_test, batch_size=64)
        # 读取一遍训练数据需要的时间
        start = time.time()
        for X, y in train_iter:
            continue
        print('%.2f sec' % (time.time() - start))  # 6.95 sec


if __name__ == "__main__":
    unittest.main()  # 运行当前源文件中的所有测试用例
