"""
    module(layer) - 常用层.

    Main members:

        # FlattenLayer - tensor展平层.
"""
from torch import nn


class FlattenLayer(nn.Module):
    """tensor展平层,返回的shape为:(batch,展平后的数值)"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)
