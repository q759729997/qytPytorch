"""
    module(cifar10) - CIFAR-10数据集常用.

    Main members:

        # get_dataset - 获取数据集.
        # get_data_iter - 获取数据集迭代器.
        # get_labels_by_ids - 根据标签id获取标签具体描述.
        # show_fashion_cifar10 - 展示图像与标签.
"""
import sys

import torch
import torchvision
from matplotlib import pyplot as plt

from qytPytorch import logger


def get_dataset(data_path, augmentation_funcs=list()):
    """ 获取数据集.

        @params:
            data_path - 数据保存路径.
            augmentation_funcs - 训练数据增强方法.

        @return:
            On success - train与test数据.
            On failure - 错误信息.
    """
    train_augmentation_funcs = [torchvision.transforms.ToTensor()]
    train_augmentation_funcs.extend(augmentation_funcs)
    augmentation_func = torchvision.transforms.Compose(train_augmentation_funcs)
    mnist_train = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=augmentation_func)
    mnist_test = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=torchvision.transforms.ToTensor())
    logger.info('dataset is :{}'.format(type(mnist_train)))
    logger.info('train data len :{}'.format(len(mnist_train)))
    logger.info('test data len :{}'.format(len(mnist_test)))
    return mnist_train, mnist_test


def get_data_iter(cifar10_train, cifar10_test, batch_size=32):
    """ 获取数据集迭代器.

        @params:
            cifar10_train - 训练数据.
            cifar10_test - 测试数据.
            batch_size - 批次大小.

        @return:
            On success - train与test数据迭代器.
            On failure - 错误信息.
    """
    if sys.platform.startswith('win'):
        num_workers = 0  # 0表示不用额外的进程来加速读取数据
    else:
        num_workers = 4
    train_iter = torch.utils.data.DataLoader(cifar10_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(cifar10_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    logger.info('train_iter len:{}'.format(len(train_iter)))
    logger.info('test_iter len:{}'.format(len(test_iter)))
    return train_iter, test_iter


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
        text_labels = ['飞机', '汽车', '鸟类', '猫', '鹿',
                       '狗', '蛙类', '马', '船', '卡车']
    else:
        text_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
    return [text_labels[int(i)] for i in label_ids]


def show_fashion_cifar10(images, labels):
    """ 展示图像与标签.

        @params:
            images - 图像特征列表.
            labels - 图像标签列表.
    """
    _, figs = plt.subplots(1, len(images), figsize=(15, 15))
    for f, img, lbl in zip(figs, images, labels):
        img = img.permute(1, 2, 0)  # 由torch.Size([3, 32, 32])转换为torch.Size([32, 32, 3])
        f.imshow(img)
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()
