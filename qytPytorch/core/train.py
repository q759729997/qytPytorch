"""
    module(train) - 模型训练.

    Main members:

        # train_net - 训练神经网络.
"""
from tqdm import tqdm

from qytPytorch import logger
from qytPytorch.metric.accuracy import evaluate_accuracy


def train_net(net, train_iter, dev_iter, max_epoch, optimizer, loss_func):
    """ 训练神经网络.

        @params:
            net - 神经网络.
            train_iter - 训练数据迭代器.
            dev_iter - 开发数据迭代器.
            max_epoch - 最大epoch.
            optimizer- 优化器.
            loss_func - 损失函数.
    """
    for epoch in range(max_epoch):
        logger.info('epoch {} begin to train'.format(epoch + 1))
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in tqdm(train_iter):
            y_hat = net(X)
            loss = loss_func(y_hat, y).sum()
            # 梯度清零
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  # “softmax回归的简洁实现”一节将用到
            train_l_sum += loss.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(dev_iter, net)
        logger.info('epoch %d, loss %.4f, train acc %.3f, test acc %.3f' % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))
