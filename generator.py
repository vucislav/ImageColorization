import torch
from torch.nn import Module, LeakyReLU, ReLU, Tanh
from layers import ConvBlock, UpsampleBlock
from functools import partial

class Generator(Module):
    def __init__(self):
        super().__init__()

        conv_block = partial(ConvBlock, kernel_sz=4, stride=2, padding=1, batch_norm=True)
        self.c1 = ConvBlock(1, 64, kernel_sz=1, batch_norm=True, act=LeakyReLU(0.2))
        self.c2 = conv_block(64, 64, act=LeakyReLU(0.2))
        self.c3 = conv_block(64, 128, act=LeakyReLU(0.2))
        self.c4 = conv_block(128, 256, act=LeakyReLU(0.2))
        self.c5 = conv_block(256, 512, act=LeakyReLU(0.2))
        self.c6 = conv_block(512, 512, act=LeakyReLU(0.2))
        self.c7 = conv_block(512, 512, act=LeakyReLU(0.2))
        self.c8 = conv_block(512, 512, act=LeakyReLU(0.2))

        up_block = partial(UpsampleBlock, kernel_sz=4, stride=2, padding=1,
            batch_norm=True, combine='concat')
        self.e1 = up_block(512, 512, act=ReLU())
        self.e2 = up_block(1024, 512, act=ReLU())
        self.e3 = up_block(1024, 512, act=ReLU())
        self.e4 = up_block(1024, 256, act=ReLU())
        self.e5 = up_block(512, 128, act=ReLU())
        self.e6 = up_block(256, 64, act=ReLU())
        self.e7 = up_block(128, 64, act=ReLU())

        self.final = ConvBlock(128, 2, kernel_sz=1, act=Tanh())
        

    def forward(self, x):
        c1 = self.c1(x)
        c2 = self.c2(c1)
        c3 = self.c3(c2)
        c4 = self.c4(c3)
        c5 = self.c5(c4)
        c6 = self.c6(c5)
        c7 = self.c7(c6)
        c8 = self.c8(c7)
        e1 = self.e1(c8, c7)
        e2 = self.e2(e1, c6)
        e3 = self.e3(e2, c5)
        e4 = self.e4(e3, c4)
        e5 = self.e5(e4, c3)
        e6 = self.e6(e5, c2)
        e7 = self.e7(e6, c1)
        final = self.final(e7)
        return final


if __name__ == '__main__':
    from torchsummary import summary
    g = Generator()
    summary(g, (1, 256, 256), device='cpu')


