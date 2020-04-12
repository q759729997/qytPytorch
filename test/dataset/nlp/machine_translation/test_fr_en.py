"""
    main_module - fr_en机器翻译常用函数，测试是将对应方法的@unittest.skip注释掉.

    Main members:

        # __main__ - 程序入口.
"""
import unittest
import sys

sys.path.insert(0, './')  # 定义搜索路径的优先顺序，序号从0开始，表示最大优先级

import qytPytorch  # noqa
print('qytPytorch module path :{}'.format(qytPytorch.__file__))  # 输出测试模块文件位置
from qytPytorch.dataset.nlp.machine_translation.fr_en import read_data  # noqa


class TestFrEn(unittest.TestCase):
    """ fr_en机器翻译常用函数.

    Main methods:
        test_read_data - 读取数据集.
    """

    # @unittest.skip('debug')
    def test_read_data(self):
        """ 读取数据集.
        """
        print('{} test_read_data {}'.format('-'*15, '-'*15))
        data_file = './data/nlp/fr-en-small.txt'
        in_vocab, out_vocab, dataset = read_data(data_file, max_seq_len=7)
        print('in_vocab len:{}, out_vocab len:{}'.format(len(in_vocab), len(out_vocab)))  # n_vocab len:46, out_vocab len:38
        print(dataset[0])  # (tensor([ 5,  4, 45,  3,  2,  0,  0]), tensor([ 8,  4, 27,  3,  2,  0,  0]))
        words = out_vocab.itos
        print('en vocab words:{}'.format(words[:20]))
        # en vocab words:['<pad>', '<bos>', '<eos>', '.', 'is', 'are', 'he', 'they', 'she', 'my', 'a', 'good', '!', 'about', 'actors', 'age', 'arguing', 'bicycle', 'both', 'brother']


if __name__ == "__main__":
    unittest.main()  # 运行当前源文件中的所有测试用例
