"""
    module(fr_en) - 机器翻译，法语和英语.

    Main members:

        # process_one_seq - 处理单个序列.
        # build_data - 使用所有的词来构造词典。并将所有序列中的词变换为词索引后构造Tensor.
        # read_data - 读取fr-en-small数据集.
"""
import codecs
import collections

import torch
import torch.utils.data as Data
import torchtext.vocab as Vocab

from qytPytorch.core.const import Const


def process_one_seq(seq_tokens, all_tokens, all_seqs, max_seq_len):
    """将一个序列中所有的词记录在all_tokens中以便之后构造词典，然后在该序列后面添加PAD直到序列,
    长度变为max_seq_len，然后将序列保存在all_seqs中.

    @params:
        seq_tokens - 单个序列tokens.
        all_tokens - 所有tokens.
        all_seqs - 所有序列（padding后）.
        max_seq_len - 最大序列长度,长度不足时进行padding.
    """
    all_tokens.extend(seq_tokens)
    seq_tokens += [Const.EOS] + [Const.PAD] * (max_seq_len - len(seq_tokens) - 1)
    all_seqs.append(seq_tokens)


def build_data(all_tokens, all_seqs):
    """ 使用所有的词来构造词典。并将所有序列中的词变换为词索引后构造Tensor.

        @params:
            all_tokens - 所有的词.
            all_seqs - 所有序列.

        @return:
            On success - vocab,转换为词索引后的Tensor.
            On failure - 错误信息.
    """
    vocab = Vocab.Vocab(collections.Counter(all_tokens),
                        specials=[Const.PAD, Const.BOS, Const.EOS])
    indices = [[vocab.stoi[w] for w in seq] for seq in all_seqs]
    return vocab, torch.tensor(indices)


def read_data(data_file, max_seq_len):
    """ 读取fr-en-small数据集.

        @params:
            data_file - 数据路径.
            max_seq_len - 最大序列长度,长度不足时进行padding.

        @return:
            On success - in_vocab,out_vocab,所有数据TensorDataset.
            On failure - 错误信息.
    """
    # in和out分别是input和output的缩写
    in_tokens, out_tokens, in_seqs, out_seqs = [], [], [], []
    with codecs.open(data_file, mode='r', encoding='utf8') as fr:
        for line in fr:
            line = line.strip()
            if len(line) == 0:
                continue
            in_seq, out_seq = line.rstrip().split('\t')
            in_seq_tokens, out_seq_tokens = in_seq.split(' '), out_seq.split(' ')
            if max(len(in_seq_tokens), len(out_seq_tokens)) > max_seq_len - 1:
                continue  # 如果加上EOS后长于max_seq_len，则忽略掉此样本
            process_one_seq(in_seq_tokens, in_tokens, in_seqs, max_seq_len)
            process_one_seq(out_seq_tokens, out_tokens, out_seqs, max_seq_len)
    in_vocab, in_data = build_data(in_tokens, in_seqs)
    out_vocab, out_data = build_data(out_tokens, out_seqs)
    return in_vocab, out_vocab, Data.TensorDataset(in_data, out_data)
