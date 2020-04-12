"""
    module(train) - 模型训练.

    Main members:

        # train_net - 训练神经网络.
        # train_machine_translation_net - 训练machine_translation神经网络.
"""
from tqdm import tqdm
import torch
from torch import nn
import torch.utils.data as Data

from qytPytorch import logger
from qytPytorch.metric.accuracy import evaluate_accuracy
from qytPytorch.metric.machine_translation import get_batch_loss


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


def train_machine_translation_net(encoder, decoder, dataset, out_vocab, lr=0.01, batch_size=8, max_epoch=5):
    """ 训练machine_translation神经网络.

        @params:
            encoder - 编码器神经网络.
            decoder - 解码器神经网络.
            dataset - 数据集.
            out_vocab - 解码器vocab.
            lr - 学习率.
            batch_size - 批次大小.
            max_epoch - 最大epoch.
    """
    enc_optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    dec_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)

    loss = nn.CrossEntropyLoss(reduction='none')
    data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)
    for epoch in range(max_epoch):
        l_sum = 0.0
        for X, Y in data_iter:
            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()
            loss_func = get_batch_loss(encoder, decoder, X, Y, loss, out_vocab=out_vocab)
            loss_func.backward()
            enc_optimizer.step()
            dec_optimizer.step()
            l_sum += loss_func.item()
        logger.info("epoch %d, loss %.3f" % (epoch + 1, l_sum / len(data_iter)))
