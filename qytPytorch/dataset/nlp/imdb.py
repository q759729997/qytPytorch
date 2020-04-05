"""
    module(fashion_mnist) - FashionMNIST数据集常用.

    Main members:

        # read_imdb - 获取数据集.
"""
import codecs
import os
import random

from tqdm import tqdm
import torch
import torch.utils.data as Data

from qytPytorch import logger


def read_imdb(folder='test', data_path="aclImdb", shuffle=False):
    """ 读取imdb数据，每个样本是一条评论及其对应的标签：1表示“正面”，0表示“负面”.

        @params:
            folder - 文件夹，test或train.
            data_path - 数据类型.
            shuffle - 是否打乱顺序.

        @return:
            On success - 数据列表,pos为1，neg为0，[(text,label),(text,label)].
            On failure - 错误信息.
    """
    imdb_data = []
    for label in ['pos', 'neg']:
        folder_name = os.path.join(data_path, folder, label)
        logger.debug('load folder:{}'.format(folder_name))
        for file in tqdm(os.listdir(folder_name)):
            with codecs.open(os.path.join(folder_name, file), mode='r', encoding='utf8') as fr:
                for raw_line in fr:
                    review = raw_line.replace('\n', '').lower()
                    if len(review) > 0:
                        imdb_data.append([review, 1 if label == 'pos' else 0])
    if shuffle:
        random.shuffle(imdb_data)
    return imdb_data


def get_tokenized_imdb(imdb_data):
    """ imdb分词.

        @params:
            imdb_data - 数据列表,pos为1，neg为0，[(text,label),(text,label)].

        @return:
            On success - 数据列表,words为分词列表,pos为1，neg为0，[(words,label),(words,label)].
            On failure - 错误信息.
    """
    def tokenize(text):
        return [tok for tok in text.split(' ')]
    return [tokenize(review.lower()) for review, _ in imdb_data]


def get_imdb_data_iter(imdb_data, vocab, padding_length=500, batch_size=32, shuffle=False):
    """ 预处理数据集,构造DataLoader.

        @params:
            imdb_data - 数据列表,pos为1，neg为0，[(text,label),(text,label)].
            vocab - 数据字典.
            padding_length - 将每条评论通过截断或者补0，使得长度变成该值.
            batch_size - 批次大小.
            shuffle - 是否打乱顺序.

        @return:
            On success - DataLoader.
            On failure - 错误信息.
    """
    def pad(x):
        return x[:padding_length] if len(x) > padding_length else x + [0] * (padding_length - len(x))
    # 分词
    tokenized_data = get_tokenized_imdb(imdb_data)
    # 单词转ID数字，进行padding
    features = torch.tensor([pad([vocab.stoi[word] for word in words]) for words in tokenized_data])
    # 标签已经转数字，因此在这里不需要进行转换
    labels = torch.tensor([label for _, label in imdb_data])
    # 构造数据集tensor迭代器
    dataset = Data.TensorDataset(features, labels)
    data_iter = Data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_iter
