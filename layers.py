import torch
from torch.nn import Module, ReLU, Conv2d, BatchNorm2d, Identity, Dropout, ConvTranspose2d

from functools import partial

class ConvBlock(Module):
    '''
    A convolutional or transpose convolutional block with optional batch 
    normalization layer, dropout and activation function.

    inc: Number of input channels
    outc: Number of output channels
    kernel_sz: Size of convolution kernel
    stride: Convolution stride
    padding: Convolution padding
    act: Activation function to be applied
    batch_norm: Include Batch Normalization layer
    dropout: Dropout zero-out probability, None if no Dropout layer
    transpose: Whether to use Convolution or Transpose Convolution
    '''
    def __init__(self, inc, outc, kernel_sz=3, stride=1, padding=0, act=ReLU(), 
        batch_norm=False, dropout=None, transpose=False):
        super().__init__()

        self.conv_layer = Conv2d(inc, outc, kernel_sz, stride=stride, padding=padding)\
            if not transpose else\
            ConvTranspose2d(inc, outc, kernel_sz, stride=stride, padding=padding)
        self.batch_norm_layer = BatchNorm2d(outc) if batch_norm else Identity()
        self.activation = act
        self.dropout_layer = Dropout(dropout) if dropout and dropout > 0 else Identity()

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.batch_norm_layer(x)
        x = self.activation(x)
        x = self.dropout_layer(x)
        return x


class UpsampleBlock(Module):
    def __init__(self, inc, outc, kernel_sz=3, stride=1, padding=0, act=ReLU(), 
        batch_norm=False, dropout=None, combine='concat'):
        super().__init__()

        self.conv_block = ConvBlock(inc, outc, kernel_sz=kernel_sz, stride=stride,
            padding=padding, act=act, batch_norm=batch_norm, dropout=dropout, transpose=True)
        self.combine = partial(torch.cat, dim=1) if combine == 'concat' else lambda x: torch.add(*x)

    def forward(self, x, skip):
        x = self.conv_block(x)
        assert x.shape[-2:] == skip.shape[-2:]
        x = self.combine((x, skip))
        return x
