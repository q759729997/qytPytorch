"""
    module(statistics_utils) - 神经网络分析统计工具类.

    Main members:

        # get_parameter_number - 统计神经网络参数个数.
"""


def get_parameter_number(net):
    """ 统计神经网络参数个数.

        @params:
            net - 神经网络.

        @return:
            On success - 字典{'total': total_num, 'trainable': trainable_num}.
            On failure - 错误信息.
    """
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'total': total_num, 'trainable': trainable_num}
