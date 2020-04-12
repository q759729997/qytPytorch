"""
    main_module - BLEU（Bilingual Evaluation Understudy）评估，测试是将对应方法的@unittest.skip注释掉.

    Main members:

        # __main__ - 程序入口.
"""
import unittest
import sys

import torch

sys.path.insert(0, './')  # 定义搜索路径的优先顺序，序号从0开始，表示最大优先级

import qytPytorch  # noqa
print('qytPytorch module path :{}'.format(qytPytorch.__file__))  # 输出测试模块文件位置
from qytPytorch.metric.bleu import get_bleu_score  # noqa


class TestBLEU(unittest.TestCase):
    """ BLEU（Bilingual Evaluation Understudy）评估.

    Main methods:
        test_get_bleu_score - 计算bleu分数.
    """

    # @unittest.skip('debug')
    def test_get_bleu_score(self):
        """ 计算bleu分数.
        """
        print('{} test_get_bleu_score {}'.format('-'*15, '-'*15))
        y_hat = list('自然语言处理')
        y = list('自然语言理解')
        print(get_bleu_score(pred_tokens=y_hat, label_tokens=y, max_gram_n=2))  # 0.8034284189446518
        print(get_bleu_score(pred_tokens=y_hat, label_tokens=y, max_gram_n=3))  # 0.7367471085977821
        print(get_bleu_score(pred_tokens=y_hat, label_tokens=y, max_gram_n=4))  # 0.6878573174731396


if __name__ == "__main__":
    unittest.main()  # 运行当前源文件中的所有测试用例
