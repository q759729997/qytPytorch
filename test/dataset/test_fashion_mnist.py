"""
    main_module - FashionMnist数据集常用函数，测试是将对应方法的@unittest.skip注释掉.

    Main members:

        # __main__ - 程序入口.
"""
import unittest
import sys

sys.path.insert(0, './')  # 定义搜索路径的优先顺序，序号从0开始，表示最大优先级

import qytPytorch  # noqa
print('qytPytorch module path :{}'.format(qytPytorch.__file__))  # 输出测试模块文件位置
from qytPytorch.dataset.fashion_mnist import get_dataset  # noqa
from qytPytorch.dataset.fashion_mnist import get_labels_by_ids  # noqa
from qytPytorch.dataset.fashion_mnist import show_fashion_mnist  # noqa


class TestFashionMnist(unittest.TestCase):
    """ FashionMnist数据集常用函数.

    Main methods:
        test_get_dataset - 获取数据集.
        test_get_labels_by_ids - 根据标签id获取标签具体描述.
        test_show_fashion_mnist - 展示图像与标签.
    """

    @unittest.skip('debug')
    def test_get_dataset(self):
        """ 获取数据集.
        """
        print('{} test_get_dataset {}'.format('-'*15, '-'*15))
        data_path = './data/FashionMNIST'
        mnist_train, mnist_test = get_dataset(data_path=data_path)
        feature, label = mnist_train[0]
        print(feature.shape, label)  # torch.Size([1, 28, 28]) 9

    @unittest.skip('debug')
    def test_get_labels_by_ids(self):
        """ 根据标签id获取标签具体描述.
        """
        print('{} test_get_labels_by_ids {}'.format('-'*15, '-'*15))
        label_ids = [1, 5, 3]
        print(get_labels_by_ids(label_ids))  # ['trouser', 'sandal', 'dress']
        print(get_labels_by_ids(label_ids, return_Chinese=True))  # ['裤子', '凉鞋', '连衣裙']

    # @unittest.skip('debug')
    def test_show_fashion_mnist(self):
        """ 展示图像与标签.
        """
        print('{} test_show_fashion_mnist {}'.format('-'*15, '-'*15))
        data_path = './data/FashionMNIST'
        mnist_train, mnist_test = get_dataset(data_path=data_path)
        feature, label = mnist_train[0]
        X, y = [], []
        for i in range(10):
            X.append(mnist_train[i][0])
            y.append(mnist_train[i][1])
        show_fashion_mnist(X, get_labels_by_ids(y))   # 直接弹出图片展示页面


if __name__ == "__main__":
    unittest.main()  # 运行当前源文件中的所有测试用例
