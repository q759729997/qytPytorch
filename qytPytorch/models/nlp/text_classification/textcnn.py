import torch
from torch import nn
from torch.nn import functional as F

from qytPytorch.modules.conv_layer import GlobalMaxPool1d


class TextCNN(nn.Module):
    """ TextCNN文本分类.

    @params:
        vocab_size - 词典大小.
        labels_size - 标签个数.
        embed_dim - embed维度大小.
        kernel_sizes - 卷积核大小.
        num_channels - 卷积核个数.
    """

    def __init__(self, vocab_size, labels_size, embed_dim=50, kernel_sizes=(1, 3, 5), num_channels=(100, 100, 100)):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels), labels_size)
        # 时序最大池化层没有权重，所以可以共用一个实例
        self.pool = GlobalMaxPool1d()
        self.convs = nn.ModuleList()  # 创建多个一维卷积层
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(in_channels=embed_dim, out_channels=c, kernel_size=k))

    def forward(self, inputs):
        # 将两个形状是(批量大小, 词数, 词向量维度)的嵌入层的输出按词向量连结
        embeddings = self.embedding(inputs)  # (batch, seq_len, embed_size)
        # 根据Conv1D要求的输入格式，将词向量维，即一维卷积层的通道维(即词向量那一维)，变换到前一维
        embeddings = embeddings.permute(0, 2, 1)
        # 对于每个一维卷积层，在时序最大池化后会得到一个形状为(批量大小, 通道大小, 1)的
        # Tensor。使用flatten函数去掉最后一维，然后在通道维上连结
        encoding = torch.cat([self.pool(F.relu(conv(embeddings))).squeeze(-1) for conv in self.convs], dim=1)
        # 应用丢弃法后使用全连接层得到输出
        outputs = self.decoder(self.dropout(encoding))
        return outputs
