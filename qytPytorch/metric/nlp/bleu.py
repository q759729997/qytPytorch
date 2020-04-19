"""
    module(bleu) - BLEU（Bilingual Evaluation Understudy）评估；主要用于机器翻译评估.

    Main members:

        # get_bleu_score - 计算bleu分数.
"""
import collections
import math


def get_bleu_score(pred_tokens, label_tokens, max_gram_n=2):
    """ 计算bleu分数.

        @params:
            pred_tokens - 预测结果，单词列表.
            label_tokens - 标签，单词列表.
            max_gram_n - 最大N元特征.

        @return:
            On success - bleu分数.
            On failure - 错误信息.
    """
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, max_gram_n + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[''.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[''.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[''.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score
