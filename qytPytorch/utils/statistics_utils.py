"""
    module(statistics_utils) - 神经网络分析统计工具类.

    Main members:

        # get_parameter_number - 统计神经网络参数个数.
"""
from qytPytorch import logger


def get_parameter_number(net):
    """ 统计神经网络参数个数.

        @params:
            net - 神经网络.

        @return:
            On success - 字典，例如 {'total': total_num, 'trainable': trainable_num}.
            On failure - 错误信息.
    """
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    logger.info(' total:{}, trainable:{}'.format(_convert_num_to_English(total_num), _convert_num_to_English(trainable_num)))
    return {'total': total_num, 'trainable': trainable_num}


def _convert_num_to_English(num):
    """数字转换为英文表述"""
    thousand = 1000
    million = thousand * 1000
    billion = million * 1000
    trillion = billion * 1000
    if num > trillion:
        return '{} Trillion'.format(round(num/trillion, 3))
    if num > billion:
        return '{} Billion'.format(round(num/billion, 3))
    if num > million:
        return '{} Million'.format(round(num/million, 3))
    if num > thousand:
        return '{} Thousand'.format(round(num/thousand, 3))
