"""
    main_module - 机器翻译Attention，测试时将对应方法的@unittest.skip注释掉.

    Main members:

        # __main__ - 程序入口.
"""
import os
import sys
import unittest

import torch

sys.path.insert(0, './')  # 定义搜索路径的优先顺序，序号从0开始，表示最大优先级

import qytPytorch  # noqa
print('qytPytorch module path :{}'.format(qytPytorch.__file__))  # 输出测试模块文件位置
from qytPytorch.models.nlp.machine_translation.attention import Encoder  # noqa
from qytPytorch.models.nlp.machine_translation.attention import Decoder  # noqa
from qytPytorch.utils.statistics_utils import get_parameter_number  # noqa
from qytPytorch.dataset.nlp.machine_translation.fr_en import read_data  # noqa
from qytPytorch.core.train import train_machine_translation_net  # noqa
from qytPytorch.utils.file_units import init_file_path  # noqa
from qytPytorch.core.vocabulary import save_vocab_words  # noqa
from qytPytorch.utils.serialize import save_serialize_obj  # noqa
from qytPytorch.utils.serialize import load_serialize_obj  # noqa
from qytPytorch.core.predict import predict_translate  # noqa


class TestAttention(unittest.TestCase):
    """机器翻译Attention.

    Main methods:
        test_get_parameter_number - 统计神经网络参数个数.
        test_train - 模型训练.
        test_net_predict - 模型预测.
    """

    @unittest.skip('debug')
    def test_get_parameter_number(self):
        """ 统计神经网络参数个数.
        """
        print('{} test_get_parameter_number {}'.format('-'*15, '-'*15))
        encoder_net = Encoder(vocab_size=100, embed_size=50, num_hiddens=64, num_layers=2, drop_prob=0.5)
        print(encoder_net)
        print(get_parameter_number(encoder_net))  # total:52.232 Thousand, trainable:52.232 Thousand
        decoder_net = Decoder(vocab_size=100, embed_size=50, num_hiddens=64, num_layers=2, attention_size=10, drop_prob=0.5)
        print(decoder_net)
        print(get_parameter_number(decoder_net))  # total:72.31 Thousand, trainable:72.31 Thousand

    @unittest.skip('debug')
    def test_train(self):
        """ 模型训练.
        """
        print('{} test_train {}'.format('-'*15, '-'*15))
        # 数据集加载
        data_file = './data/nlp/fr-en-small.txt'
        in_vocab, out_vocab, dataset = read_data(data_file, max_seq_len=7)
        print('in_vocab len:{}, out_vocab len:{}'.format(len(in_vocab), len(out_vocab)))  # n_vocab len:46, out_vocab len:38
        # 构造模型
        encoder_net = Encoder(vocab_size=len(in_vocab), embed_size=50, num_hiddens=64, num_layers=2, drop_prob=0.5)
        print('encoder_net 参数量:{}'.format(get_parameter_number(encoder_net)))  # total:49.532 Thousand, trainable:49.532 Thousand
        decoder_net = Decoder(vocab_size=len(out_vocab), embed_size=50, num_hiddens=64, num_layers=2, attention_size=10, drop_prob=0.5)
        print('decoder_net 参数量:{}'.format(get_parameter_number(decoder_net)))  # total:65.18 Thousand, trainable:65.18 Thousand
        # 训练
        train_machine_translation_net(encoder=encoder_net, decoder=decoder_net, dataset=dataset, out_vocab=out_vocab, max_epoch=100)
        # 2020-04-12 16:48:01 I [train.py:74] epoch 100, loss 0.008
        # 保存模型
        file_path = './data/save/machine_translation/attention'
        init_file_path(file_path)
        torch.save(encoder_net, f=os.path.join(file_path, 'encoder_net.pkl'))

        torch.save(decoder_net, f=os.path.join(file_path, 'decoder_net.pkl'))
        # 保存vocabulary
        save_vocab_words(in_vocab, file_name=os.path.join(file_path, 'in_vocab_words.txt'))
        save_serialize_obj(in_vocab, filename=os.path.join(file_path, 'in_vocab.pkl'))

        save_vocab_words(out_vocab, file_name=os.path.join(file_path, 'out_vocab_words.txt'))
        save_serialize_obj(out_vocab, filename=os.path.join(file_path, 'out_vocab.pkl'))

    # @unittest.skip('debug')
    def test_net_predict(self):
        """ 模型预测.
        """
        print('{} test_net_predict {}'.format('-'*15, '-'*15))
        # 加载模型
        file_path = './data/save/machine_translation/attention'
        encoder_net = torch.load(os.path.join(file_path, 'encoder_net.pkl'))
        decoder_net = torch.load(os.path.join(file_path, 'decoder_net.pkl'))
        # 加载词典
        in_vocab = load_serialize_obj(os.path.join(file_path, 'in_vocab.pkl'))
        out_vocab = load_serialize_obj(os.path.join(file_path, 'out_vocab.pkl'))
        print('in_vocab len:{}, out_vocab len:{}'.format(len(in_vocab), len(out_vocab)))  # n_vocab len:46, out_vocab len:38
        # 预测
        input_seq = 'ils regardent .'
        output_tokens = predict_translate(encoder=encoder_net, decoder=decoder_net, input_seq=input_seq, max_seq_len=7, in_vocab=in_vocab, out_vocab=out_vocab)
        print(output_tokens)  # ['they', 'are', 'watching', '.']


if __name__ == "__main__":
    unittest.main()  # 运行当前源文件中的所有测试用例
