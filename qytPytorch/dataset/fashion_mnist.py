"""
    module(fashion_mnist) - FashionMNIST数据集常用.

    Main members:

        # get_dataset - 获取数据集.
        # get_labels_by_ids - 根据标签id获取标签具体描述.
        # show_fashion_mnist - 展示图像与标签.
"""
import torchvision
from matplotlib import pyplot as plt

from qytPytorch import logger


def get_dataset(data_path):
    """ 获取数据集.

        @params:
            data_path - 数据保存路径.

        @return:
            On success - train与test数据.
            On failure - 错误信息.
    """
    mnist_train = torchvision.datasets.FashionMNIST(root=data_path, train=True, download=True, transform=torchvision.transforms.ToTensor())
    mnist_test = torchvision.datasets.FashionMNIST(root=data_path, train=False, download=True, transform=torchvision.transforms.ToTensor())
    logger.info('dataset is :{}'.format(type(mnist_train)))
    logger.info('train data len :{}'.format(len(mnist_train)))
    logger.info('test data len :{}'.format(len(mnist_test)))
    return mnist_train, mnist_test


def get_labels_by_ids(label_ids, return_Chinese=False):
    """ 根据标签id获取标签具体描述.

        @params:
            label_ids - 标签id列表.
            return_Chinese - 是否返回中文.

        @return:
            On success - 转换后的标签列表.
            On failure - 错误信息.
    """
    if return_Chinese:
        text_labels = ['T恤', '裤子', '套衫', '连衣裙', '外套',
                       '凉鞋', '衬衫', '运动鞋', '包', '短靴']
    else:
        text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                       'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in label_ids]


def show_fashion_mnist(images, labels):
    """ 展示图像与标签.

        @params:
            images - 图像特征列表.
            labels - 图像标签列表.
    """
    # use_svg_display()
    # _表示我们忽略（不使用）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()
