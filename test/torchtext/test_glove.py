"""
    main_module - torchtext工具包，测试时将对应方法的@unittest.skip注释掉.

    Main members:

        # __main__ - 程序入口.
"""
import sys
import unittest

import torchtext

sys.path.insert(0, './')  # 定义搜索路径的优先顺序，序号从0开始，表示最大优先级

import qytPytorch  # noqa
print('qytPytorch module path :{}'.format(qytPytorch.__file__))  # 输出测试模块文件位置
from qytPytorch.core.embedding import load_pretrained_embedding  # noqa


class TestTorchtext(unittest.TestCase):
    """torchtext工具包.

    Main methods:
        test_pretrained_aliases - 目前提供的预训练词嵌入的名称.
        test_pretrained_glove - 预训练glove模型测试.
        test_load_pretrained_embedding - 从预训练好的vocab中提取出words对应的词向量.
    """
    @unittest.skip('debug')
    def test_pretrained_aliases(self):
        """目前提供的预训练词嵌入的名称.
        """
        print('{} test_pretrained_aliases {}'.format('-'*15, '-'*15))
        print(torchtext.vocab.pretrained_aliases.keys())
        # dict_keys(['charngram.100d', 'fasttext.en.300d', 'fasttext.simple.300d', 'glove.42B.300d', 'glove.840B.300d', 'glove.twitter.27B.25d', 'glove.twitter.27B.50d', 'glove.twitter.27B.100d', 'glove.twitter.27B.200d', 'glove.6B.50d', 'glove.6B.100d', 'glove.6B.200d', 'glove.6B.300d'])

    # @unittest.skip('debug')
    def test_pretrained_glove(self):
        """预训练glove模型测试.
        """
        print('{} test_pretrained_glove {}'.format('-'*15, '-'*15))
        glove = torchtext.vocab.GloVe(name='6B', dim=50, cache='./data/torchtext')
        print("一共包含%d个词。" % len(glove.stoi))  # 一共包含400000个词。
        print(glove.stoi['beautiful'], glove.itos[3366])  # 3366 beautiful

    @unittest.skip('debug')
    def test_load_pretrained_embedding(self):
        """从预训练好的vocab中提取出words对应的词向量.
        """
        print('{} test_load_pretrained_embedding {}'.format('-'*15, '-'*15))
        glove_embedding = torchtext.vocab.GloVe(name='6B', dim=50, cache='./data/torchtext')
        print("一共包含%d个词。" % len(glove_embedding.stoi))  # 一共包含400000个词。
        words = ['i', 'love', 'you']
        embed = load_pretrained_embedding(words=words, pretrained_vocab=glove_embedding)
        print(embed.shape)  # torch.Size([3, 50])
        print(embed)


if __name__ == "__main__":
    unittest.main()  # 运行当前源文件中的所有测试用例
