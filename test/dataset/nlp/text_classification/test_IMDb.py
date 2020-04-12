"""
    main_module - IMDb情感分类常用函数，测试是将对应方法的@unittest.skip注释掉.

    Main members:

        # __main__ - 程序入口.
"""
import os
import unittest
import sys

sys.path.insert(0, './')  # 定义搜索路径的优先顺序，序号从0开始，表示最大优先级

import qytPytorch  # noqa
print('qytPytorch module path :{}'.format(qytPytorch.__file__))  # 输出测试模块文件位置
from qytPytorch.utils.file_units import extract_tarfile  # noqa
from qytPytorch.dataset.nlp.text_classification.imdb import read_imdb  # noqa
from qytPytorch.dataset.nlp.text_classification.imdb import get_tokenized_imdb  # noqa
from qytPytorch.dataset.nlp.text_classification.imdb import get_imdb_data_iter  # noqa
from qytPytorch.utils.serialize import save_serialize_obj  # noqa
from qytPytorch.utils.serialize import load_serialize_obj  # noqa
from qytPytorch.core.vocabulary import get_tokenized_vocab  # noqa


class TestIMDb(unittest.TestCase):
    """ IMDb情感分类常用函数.

    Main methods:
        test_get_dataset - 获取数据集.
        test_read_imdb - 读取数据集.
        test_get_tokenized_imdb - 数据集分词.
        test_tokenized_vocab - 数据集词典构造.
        test_get_imdb_data_iter - 预处理数据集,构造DataLoader.
    """

    @unittest.skip('debug')
    def test_get_dataset(self):
        """ 获取数据集.
        下载链接(压缩包越80MB)： http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
        """
        print('{} test_get_dataset {}'.format('-'*15, '-'*15))
        # 数据解压，耗时特别长
        data_path = './data'
        tarfile_name = os.path.join(data_path, 'aclImdb_v1.tar.gz')
        print(extract_tarfile(tarfile_name, output_path=data_path, target_file_name='aclImdb'))

    @unittest.skip('debug')
    def test_read_imdb(self):
        """ 读取数据集.
        """
        print('{} test_read_imdb {}'.format('-'*15, '-'*15))
        data_path = './data/aclImdb'
        test_data_pickle = './data/aclImdb/test_data.pkl'
        test_data = read_imdb(folder='test', data_path=data_path)
        print('test_data len:{}'.format(len(test_data)))  # test_data len:25737
        print('example:{}'.format(test_data[:3]))
        # 数据集序列化，方便直接加载
        save_serialize_obj(test_data, test_data_pickle)

    @unittest.skip('debug')
    def test_get_tokenized_imdb(self):
        """ 数据集分词.
        """
        print('{} test_get_tokenized_imdb {}'.format('-'*15, '-'*15))
        test_data_pickle = './data/aclImdb/test_data.pkl'
        test_data_tokenized_pickle = './data/aclImdb/test_data_tokenized.pkl'
        test_data = load_serialize_obj(test_data_pickle)
        test_data_tokenized = get_tokenized_imdb(imdb_data=test_data)
        print('test_data_tokenized len:{}'.format(len(test_data_tokenized)))  # test_data_tokenized len:25737
        print('example:{}'.format(test_data_tokenized[:3]))
        # 数据集序列化，方便直接加载
        save_serialize_obj(test_data_tokenized, test_data_tokenized_pickle)

    @unittest.skip('debug')
    def test_tokenized_vocab(self):
        """ 数据集词典构造.
        """
        print('{} test_tokenized_vocab {}'.format('-'*15, '-'*15))
        test_data_tokenized_pickle = './data/aclImdb/test_data_tokenized.pkl'
        test_data_tokenized = load_serialize_obj(test_data_tokenized_pickle)
        test_data_vocab = get_tokenized_vocab(test_data_tokenized)
        print('vocab len:{}'.format(len(test_data_vocab)))  # vocab len:45098
        print('overcome id:{}'.format(test_data_vocab.stoi.get('overcome', None)))  # overcome id:3753
        print('father id:{}'.format(test_data_vocab.stoi.get('father', None)))  # father id:475

    @unittest.skip('debug')
    def test_get_imdb_data_iter(self):
        """ 预处理数据集,构造DataLoader.
        """
        print('{} test_get_imdb_data_iter {}'.format('-'*15, '-'*15))
        test_data_pickle = './data/aclImdb/test_data.pkl'
        test_data = load_serialize_obj(test_data_pickle)
        test_data_tokenized = get_tokenized_imdb(imdb_data=test_data)
        test_data_vocab = get_tokenized_vocab(test_data_tokenized)
        print('vocab len:{}'.format(len(test_data_vocab)))  # vocab len:45098
        test_iter = get_imdb_data_iter(test_data, test_data_vocab, batch_size=8, shuffle=True)
        print('test_iter len:{}'.format(len(test_iter)))  # test_iter len:3218
        for X, y in test_iter:
            print('X', X.shape, 'y', y.shape)  # X torch.Size([8, 500]) y torch.Size([8])
            break


if __name__ == "__main__":
    unittest.main()  # 运行当前源文件中的所有测试用例
