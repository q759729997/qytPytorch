"""
    module(accuracy) - 准确率评估.

    Main members:

        # get_accuracy_score - 计算准确率分数.
        # evaluate_accuracy - 准确率评估，模型预测并评估.
"""


def get_accuracy_score(y_hat, y):
    """ 计算准确率分数.

        @params:
            y_hat - 预测结果.
            y - 真实结果.

        @return:
            On success - 准确率分数.
            On failure - 错误信息.
    """
    return (y_hat.argmax(dim=1) == y).float().mean().item()


def evaluate_accuracy(data_iter, net):
    """ 准确率评估，模型预测并评估.

        @params:
            data_iter - 数据集迭代器.
            net - 神经网络.

        @return:
            On success - 准确率分数.
            On failure - 错误信息.
    """
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n
