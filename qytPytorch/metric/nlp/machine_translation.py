"""
    module(machine_translation) - 机器翻译常用评估函数.

    Main members:

        # get_batch_loss - 机器翻译批量损失.
"""
import torch

from qytPytorch.core.const import Const


def get_batch_loss(encoder, decoder, X, Y, loss_func, out_vocab):
    """ 机器翻译批量损失.

        @params:
            encoder - 编码器.
            decoder - 解码器.
            X - 输入tensor.
            Y - 输出tensor.
            loss_func - 损失函数.
            out_vocab - 解码vocab.

        @return:
            On success - 损失值.
            On failure - 错误信息.
    """
    batch_size = X.shape[0]
    enc_state = encoder.begin_state()
    enc_outputs, enc_state = encoder(X, enc_state)
    # 初始化解码器的隐藏状态
    dec_state = decoder.begin_state(enc_state)
    # 解码器在最初时间步的输入是BOS
    dec_input = torch.tensor([out_vocab.stoi[Const.BOS]] * batch_size)
    # 我们将使用掩码变量mask来忽略掉标签为填充项PAD的损失
    mask, num_not_pad_tokens = torch.ones(batch_size,), 0
    loss_value = torch.tensor([0.0])
    for y in Y.permute(1, 0):  # Y shape: (batch, seq_len)
        dec_output, dec_state = decoder(dec_input, dec_state, enc_outputs)
        loss_value = loss_value + (mask * loss_func(dec_output, y)).sum()
        dec_input = y  # 使用强制教学
        num_not_pad_tokens += mask.sum().item()
        # EOS后面全是PAD. 下面一行保证一旦遇到EOS接下来的循环中mask就一直是0
        mask = mask * (y != out_vocab.stoi[Const.EOS]).float()
    return loss_value / num_not_pad_tokens
