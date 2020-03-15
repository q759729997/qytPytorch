"""
    module(accuracy) - 准确率评估.

    Main members:

        # get_accuracy_score - 计算准确率分数.
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
