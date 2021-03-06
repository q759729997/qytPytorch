"""
    main_module - 文本分类BiLSTM，测试时将对应方法的@unittest.skip注释掉.

    Main members:

        # __main__ - 程序入口.
"""
import sys
import unittest

import torch
import torchtext
from torch import nn

sys.path.insert(0, './')  # 定义搜索路径的优先顺序，序号从0开始，表示最大优先级

import qytPytorch  # noqa
print('qytPytorch module path :{}'.format(qytPytorch.__file__))  # 输出测试模块文件位置
from qytPytorch.models.nlp.text_classify import BiLSTM  # noqa
from qytPytorch.utils.statistics_utils import get_parameter_number  # noqa
from qytPytorch.utils.serialize import save_serialize_obj  # noqa
from qytPytorch.utils.serialize import load_serialize_obj  # noqa
from qytPytorch.core.vocabulary import get_tokenized_vocab  # noqa
from qytPytorch.core.vocabulary import save_vocab_words  # noqa
from qytPytorch.dataset.nlp.imdb import get_tokenized_imdb  # noqa
from qytPytorch.dataset.nlp.imdb import get_imdb_data_iter  # noqa
from qytPytorch.dataset.nlp.imdb import predict_sentiment  # noqa
from qytPytorch.core.train import train_net  # noqa
from qytPytorch.core.embedding import load_pretrained_embedding  # noqa


class TestBiLSTM(unittest.TestCase):
    """BiLSTM.

    Main methods:
        test_get_out_shape - 计算神经网络输出形状.
        test_get_parameter_number - 统计神经网络参数个数.
        test_train - 模型训练.
        test_train_use_pretrained_embedding - 模型训练,使用预训练embed.
        test_net_predict - 模型预测.
    """
    # @unittest.skip('debug')
    def test_get_out_shape(self):
        """计算神经网络输出形状.
        """
        print('{} test_get_out_shape {}'.format('-'*15, '-'*15))
        input_shape = (8, 50)  # 批量大小,句子长度
        net = BiLSTM(vocab_size=1000, labels_size=2)
        # print(net)
        x = torch.randint(low=1, high=500, size=input_shape, dtype=torch.long)  # 数据类型需要为long，以便于embed进行转换
        y = net(x)
        print(y.shape)  # torch.Size([8, 2])
        print(y)
        """
        tensor([[ 0.0216,  0.0019],
        [ 0.0485,  0.0487],
        [ 0.0136, -0.0197],
        [ 0.0255, -0.0421],
        [ 0.0079, -0.0233],
        [ 0.0694,  0.0226],
        [ 0.0136, -0.0249],
        [-0.0350, -0.0268]], grad_fn=<AddmmBackward>)
        """

    @unittest.skip('debug')
    def test_get_parameter_number(self):
        """ 统计神经网络参数个数.
        """
        print('{} test_get_parameter_number {}'.format('-'*15, '-'*15))
        net = BiLSTM(vocab_size=100000, labels_size=2)
        print(net)
        print(get_parameter_number(net))  # {'total': 454802, 'trainable': 454802}
        # vocab_size=1000   时 {'total': 454802, 'trainable': 454802}
        # vocab_size=100000 时 {'total': 5404802, 'trainable': 5404802}

    @unittest.skip('debug')
    def test_train(self):
        """ 模型训练.
        """
        print('{} test_train {}'.format('-'*15, '-'*15))
        # 数据集加载
        test_data_pickle = './data/aclImdb/test_data.pkl'
        test_data = load_serialize_obj(test_data_pickle)
        test_data = test_data[:100]  # 数据量比较大，cpu电脑跑不动，取一部分进行训练
        test_data_tokenized = get_tokenized_imdb(imdb_data=test_data)
        test_data_vocab = get_tokenized_vocab(test_data_tokenized)
        vocab_size = len(test_data_vocab)
        print('vocab len:{}'.format(vocab_size))  # vocab len:45098
        test_iter = get_imdb_data_iter(test_data, test_data_vocab, batch_size=8, shuffle=True)
        print('test_iter len:{}'.format(len(test_iter)))  # test_iter len:3218
        # 构造模型
        net = BiLSTM(vocab_size=vocab_size, labels_size=2)
        print('参数量:{}'.format(get_parameter_number(net)))  # total:436.002 Thousand, trainable:436.002 Thousand
        print(net)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()))
        loss_func = nn.CrossEntropyLoss()
        # 训练
        train_net(net, train_iter=test_iter, dev_iter=test_iter, max_epoch=5, optimizer=optimizer, loss_func=loss_func)

    @unittest.skip('debug')
    def test_train_use_pretrained_embedding(self):
        """ 模型训练,使用预训练embed.
        """
        print('{} test_train_use_pretrained_embedding {}'.format('-'*15, '-'*15))
        # 数据集加载
        test_data_pickle = './data/aclImdb/test_data.pkl'
        test_data = load_serialize_obj(test_data_pickle)
        test_data = test_data[:1000]  # 数据量比较大，cpu电脑跑不动，取一部分进行训练
        test_data_tokenized = get_tokenized_imdb(imdb_data=test_data)
        test_data_vocab = get_tokenized_vocab(test_data_tokenized)
        vocab_size = len(test_data_vocab)
        print('vocab len:{}'.format(vocab_size))  # vocab len:45098
        test_iter = get_imdb_data_iter(test_data, test_data_vocab, batch_size=8, shuffle=True)
        print('test_iter len:{}'.format(len(test_iter)))  # test_iter len:3218
        # 构造模型
        net = BiLSTM(vocab_size=vocab_size, labels_size=2)
        print('参数量:{}'.format(get_parameter_number(net)))  # total:436.002 Thousand, trainable:436.002 Thousand
        # 使用预训练embed初始化
        glove_embedding = torchtext.vocab.GloVe(name='6B', dim=50, cache='./data/torchtext')
        print("glove_embedding 一共包含%d个词。" % len(glove_embedding.stoi))  # 一共包含400000个词。
        words = test_data_vocab.itos
        embed = load_pretrained_embedding(words=words, pretrained_vocab=glove_embedding)  # There are 73 oov words.
        net.embedding.weight.data.copy_(embed)
        net.embedding.weight.requires_grad = False  # 直接加载预训练好的, 所以不需要更新它
        print('参数量:{}'.format(get_parameter_number(net)))  # total:436.002 Thousand, trainable:404.802 Thousand
        print(net)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()))
        loss_func = nn.CrossEntropyLoss()
        # 训练
        train_net(net, train_iter=test_iter, dev_iter=test_iter, max_epoch=2, optimizer=optimizer, loss_func=loss_func)
        # 保存模型
        torch.save(net, f='./data/save/text_classify/bilstm/model.pkl')
        # 保存vocabulary
        save_vocab_words(test_data_vocab, file_name='./data/save/text_classify/bilstm/vocab_words.txt')
        save_serialize_obj(test_data_vocab, filename='./data/save/text_classify/bilstm/vocab.pkl')

    @unittest.skip('debug')
    def test_net_predict(self):
        """ 模型预测.
        """
        print('{} test_net_predict {}'.format('-'*15, '-'*15))
        # 加载模型
        model_file = './data/save/text_classify/bilstm/model.pkl'
        net = torch.load(model_file)
        print(net)
        # 加载词典
        vocab_file = './data/save/text_classify/bilstm/vocab.pkl'
        vocab_obj = load_serialize_obj(vocab_file)
        print('vocab len:{}'.format(len(vocab_obj)))  # vocab len:624
        # 预测
        label = predict_sentiment(net=net, vocab_obj=vocab_obj, words=['this', 'movie', 'is', 'so', 'good'])
        print(label)
        label = predict_sentiment(net=net, vocab_obj=vocab_obj, words=['terrible', 'movie', 'is', 'so', 'bad'])
        print(label)


if __name__ == "__main__":
    unittest.main()  # 运行当前源文件中的所有测试用例
