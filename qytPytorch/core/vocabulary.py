"""
    module(vocabulary) - 词典模块.

    Main members:

        # get_tokenized_vocab - 构造词典.
"""
import collections
import torchtext.vocab as Vocab


def get_tokenized_vocab(tokenized_data, min_freq=5):
    """ 构造词典.

        @params:
            tokenized_data - 分词后的数据,words为单词列表，[[words], [words]].
            min_freq - 构造Vocab时的最小词频.

        @return:
            On success - Vocab对象.
            On failure - 错误信息.
    """
    tokenized_counter = collections.Counter([tk for st in tokenized_data for tk in st])
    return Vocab.Vocab(tokenized_counter, min_freq=5)
