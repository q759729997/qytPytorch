"""
    module(embedding) - 词嵌入.

    Main members:

        # load_pretrained_embedding - 从预训练好的vocab中提取出words对应的词向量.
"""
import torch

from qytPytorch import logger


def load_pretrained_embedding(words, pretrained_vocab):
    """ 从预训练好的vocab中提取出words对应的词向量.

        @params:
            words - 单词列表.
            pretrained_vocab - 预训练embedding.

        @return:
            On success - words对应的embed,tensor.
            On failure - 错误信息.
    """
    embed = torch.zeros(len(words), pretrained_vocab.vectors[0].shape[0]) # 初始化为0
    oov_count = 0  # out of vocabulary
    for i, word in enumerate(words):
        try:
            idx = pretrained_vocab.stoi[word]
            embed[i, :] = pretrained_vocab.vectors[idx]
        except KeyError:
            oov_count += 1
    if oov_count > 0:
        logger.warn("There are %d oov words." % oov_count)
    return embed
