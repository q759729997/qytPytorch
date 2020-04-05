"""
    module(vocabulary) - 词典模块.

    Main members:

        # get_tokenized_vocab - 构造词典.
        # save_vocab_words - 保存词典中的单词与单词id.
"""
import codecs

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


def save_vocab_words(vocab_obj, file_name):
    """ 保存词典中的单词与单词id.

        @params:
            vocab_obj - Vocab对象.
            file_name - 文件名称.

        @return:
            On success - Vocab对象.
            On failure - 错误信息.
    """
    with codecs.open(file_name, mode='w', encoding='utf8') as fw:
        for word in vocab_obj.itos:
            word_id = vocab_obj.stoi[word]
            fw.write('{}\t{}\n'.format(word, word_id))
