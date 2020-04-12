"""
    module(train) - 模型预测.

    Main members:

        # predict_translate - 机器翻译预测.
"""
import torch

from qytPytorch.core.const import Const


def predict_translate(encoder, decoder, input_seq, max_seq_len, in_vocab, out_vocab):
    """ 机器翻译预测.

        @params:
            encoder - 编码器模型.
            decoder - 解码器模型.
            input_seq - 输入序列.
            max_seq_len - 最大序列长度,长度不足时进行padding.
            in_vocab - 输入vocab.
            out_vocab - 输出vocab.

        @return:
            On success - 预测结果.
            On failure - 错误信息.
    """
    in_tokens = input_seq.split(' ')
    in_tokens += [Const.EOS] + [Const.PAD] * (max_seq_len - len(in_tokens) - 1)
    enc_input = torch.tensor([[in_vocab.stoi[tk] for tk in in_tokens]])  # batch=1
    enc_state = encoder.begin_state()
    enc_output, enc_state = encoder(enc_input, enc_state)
    dec_input = torch.tensor([out_vocab.stoi[Const.BOS]])
    dec_state = decoder.begin_state(enc_state)
    output_tokens = []
    for _ in range(max_seq_len):
        dec_output, dec_state = decoder(dec_input, dec_state, enc_output)
        pred = dec_output.argmax(dim=1)
        pred_token = out_vocab.itos[int(pred.item())]
        if pred_token == Const.EOS:  # 当任一时间步搜索出EOS时，输出序列即完成
            break
        else:
            output_tokens.append(pred_token)
            dec_input = pred
    return output_tokens
