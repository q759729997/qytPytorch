import torch
from torch import nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """ 编码器.

    @params:
        vocab_size - 词典大小.
        embed_size - 词向量维度.
        num_hiddens - 隐层个数.
        num_layers - 隐层层数.
        drop_prob - drop概率.
    """
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 drop_prob=0, **kwargs):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=drop_prob)

    def forward(self, inputs, state):
        # 输入形状是(批量大小, 时间步数)。将输出互换样本维和时间步维
        # 对于门控循环单元来说，state就是一个元素，即隐藏状态；如果使用长短期记忆，state是一个元组，包含两个元素即隐藏状态和记忆细胞。
        embedding = self.embedding(inputs.long()).permute(1, 0, 2)  # (seq_len, batch, input_size)
        return self.rnn(embedding, state)

    def begin_state(self):
        """隐藏态初始化为None时PyTorch会自动初始化为0"""
        return None


class Decoder(nn.Module):
    """ 解码器.
    由于解码器的输入来自输出语言的词索引，我们将输入通过词嵌入层得到表征，然后和背景向量在特征维连结。我们将连结后的结果与上一时间步的隐藏状态通过门控循环单元计算出当前时间步的输出与隐藏状态。最后，我们将输出通过全连接层变换为有关各个输出词的预测，形状为(批量大小, 输出词典大小)。

    @params:
        vocab_size - 词典大小.
        embed_size - 词向量维度.
        num_hiddens - 隐层个数.
        num_layers - 隐层层数.
        attention_size - attention_size.
        drop_prob - drop概率.
    """
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 attention_size, drop_prob=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = self._attention_model(2*num_hiddens, attention_size)
        # GRU的输入包含attention输出的c和实际输入, 所以尺寸是 num_hiddens+embed_size
        self.rnn = nn.GRU(num_hiddens + embed_size, num_hiddens, 
                          num_layers, dropout=drop_prob)
        self.out = nn.Linear(num_hiddens, vocab_size)

    def forward(self, cur_input, state, enc_states):
        """
        cur_input shape: (batch, )
        state shape: (num_layers, batch, num_hiddens)
        """
        # 使用注意力机制计算背景向量
        c = self._attention_forward(self.attention, enc_states, state[-1])
        # 将嵌入后的输入和背景向量在特征维连结, (批量大小, num_hiddens+embed_size)
        input_and_c = torch.cat((self.embedding(cur_input), c), dim=1)
        # 为输入和背景向量的连结增加时间步维，时间步个数为1
        output, state = self.rnn(input_and_c.unsqueeze(0), state)
        # 移除时间步维，输出形状为(批量大小, 输出词典大小)
        output = self.out(output).squeeze(dim=0)
        return output, state

    def begin_state(self, enc_state):
        # 直接将编码器最终时间步的隐藏状态作为解码器的初始隐藏状态
        return enc_state

    def _attention_model(self, input_size, attention_size):
        """注意力模型"""
        # 定义的函数aa：将输入连结后通过含单隐藏层的多层感知机变换。其中隐藏层的输入是解码器的隐藏状态与编码器在所有时间步上隐藏状态的一一连结，且使用tanh函数作为激活函数。输出层的输出个数为1。两个Linear实例均不使用偏差。
        model = nn.Sequential(nn.Linear(input_size, attention_size, bias=False),
                              nn.Tanh(),
                              nn.Linear(attention_size, 1, bias=False))
        return model

    def _attention_forward(self, model, enc_states, dec_state):
        """
        enc_states: (时间步数, 批量大小, 隐藏单元个数)
        dec_state: (批量大小, 隐藏单元个数)
        """
        # 注意力机制的输入包括查询项、键项和值项。设编码器和解码器的隐藏单元个数相同。这里的查询项为解码器在上一时间步的隐藏状态，形状为(批量大小, 隐藏单元个数)；键项和值项均为编码器在所有时间步的隐藏状态，形状为(时间步数, 批量大小, 隐藏单元个数)。注意力机制返回当前时间步的背景变量，形状为(批量大小, 隐藏单元个数)。
        # 将解码器隐藏状态广播到和编码器隐藏状态形状相同后进行连结
        dec_states = dec_state.unsqueeze(dim=0).expand_as(enc_states)
        enc_and_dec_states = torch.cat((enc_states, dec_states), dim=2)
        e = model(enc_and_dec_states)  # 形状为(时间步数, 批量大小, 1)
        alpha = F.softmax(e, dim=0)  # 在时间步维度做softmax运算
        return (alpha * enc_states).sum(dim=0)  # 返回背景变量
