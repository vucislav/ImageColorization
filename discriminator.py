import torch
from torch.nn import Module, LeakyReLU, Sigmoid
from layers import ConvBlock
from functools import partial

class Discriminator(Module):
    def __init__(self):
        super().__init__()

        conv_block = partial(ConvBlock, kernel_sz=4, stride=2, padding=1, batch_norm=True)
        self.c1 = ConvBlock(3, 64, kernel_sz=1, batch_norm=True, act=LeakyReLU(0.2))
        self.c2 = conv_block(64, 64, act=LeakyReLU(0.2))
        self.c3 = conv_block(64, 128, act=LeakyReLU(0.2))
        self.c4 = conv_block(128, 256, act=LeakyReLU(0.2))
        self.c5 = conv_block(256, 512, act=LeakyReLU(0.2))
        self.c6 = conv_block(512, 512, act=LeakyReLU(0.2))
        self.c7 = conv_block(512, 512, act=LeakyReLU(0.2))
        self.c8 = conv_block(512, 512, act=LeakyReLU(0.2))

        self.final = ConvBlock(512, 1, kernel_sz=2, act=Sigmoid())
        

    def forward(self, x):
        c1 = self.c1(x)
        c2 = self.c2(c1)
        c3 = self.c3(c2)
        c4 = self.c4(c3)
        c5 = self.c5(c4)
        c6 = self.c6(c5)
        c7 = self.c7(c6)
        c8 = self.c8(c7)
        final = self.final(c8)
        return final


if __name__ == '__main__':
    from torchsummary import summary
    g = Discriminator()
    summary(g, (1, 256, 256), device='cpu')


