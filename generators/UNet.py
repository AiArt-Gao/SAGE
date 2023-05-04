import torch
import torch.nn as nn
from torch.nn import init
from generators.spadeDecoder import SPADEResnetBlock


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(ch_out),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(ch_out),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class spade_conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(spade_conv_block, self).__init__()
        # self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True)
        self.spade1 = SPADEResnetBlock(ch_in, 19)
        self.activate1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True)
        self.spade2 = SPADEResnetBlock(ch_out, 19)
        self.activate2 = nn.LeakyReLU(0.2, inplace=True)


    def forward(self, x, parsing):
        # x = self.conv1(x)
        x = self.spade1(x, parsing)
        x = self.activate1(x)
        x = self.conv(x)
        x = self.spade2(x, parsing)
        x = self.activate2(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(ch_out),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class U_Net(nn.Module):
    def __init__(self, input_ch=3, output_ch=3):
        super(U_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(ch_in=input_ch, ch_out=32)
        self.Conv2 = conv_block(ch_in=32, ch_out=64)
        self.Conv3 = conv_block(ch_in=64, ch_out=128)
        self.Conv4 = conv_block(ch_in=128, ch_out=256)
        self.Conv5 = conv_block(ch_in=256, ch_out=512)

        self.Up5 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv5 = spade_conv_block(ch_in=512, ch_out=256)

        self.Up4 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv4 = spade_conv_block(ch_in=256, ch_out=128)

        self.Up3 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv3 = conv_block(ch_in=128, ch_out=64)

        self.Up2 = up_conv(ch_in=64, ch_out=32)
        self.Up_conv2 = conv_block(ch_in=64, ch_out=32)

        self.Conv_1x1 = nn.Conv2d(32, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x, parsing):
        """
        :param x: input RGB image [b, 3, 256, 256]
        :return: RGB image [b, 3, 256 ,256]
        """
        # encoding path

        x1 = self.Conv1(x)      # x1: [b, 64, 256, 256]

        x2 = self.Maxpool(x1)   # x2: [b, 64, 128, 128]
        x2 = self.Conv2(x2)     # x2: [b, 128, 128, 128]

        x3 = self.Maxpool(x2)   # x3: [b, 128, 64, 64]
        x3 = self.Conv3(x3)     # x3: [b, 256, 64, 64]

        x4 = self.Maxpool(x3)   # x4: [b, 256, 32, 32]
        x4 = self.Conv4(x4)     # x4: [b, 512, 32, 32]

        x5 = self.Maxpool(x4)   # x5: [b, 512, 16, 16]
        x5 = self.Conv5(x5)     # x5: [b, 1024 ,16, 16]

        # decoding + concat path
        d5 = self.Up5(x5)       # d5: [b, 512, 32, 32]
        d5 = torch.cat((x4, d5), dim=1)     # d5: [b, 1024, 32, 32]

        d5 = self.Up_conv5(d5, parsing)      # d5: [b, 512, 32, 32]

        d4 = self.Up4(d5)       # d4: [b, 256, 64, 64]
        d4 = torch.cat((x3, d4), dim=1)     # d4:: [b, 512, 64, 64]
        d4 = self.Up_conv4(d4, parsing)      # d4: [b, 256, 64, 64]

        d3 = self.Up3(d4)       # d3: [b, 128, 128, 128]
        d3 = torch.cat((x2, d3), dim=1)     # d3: [b, 256, 128, 128]
        d3 = self.Up_conv3(d3)      # d3: [b, 128, 128, 128]

        d2 = self.Up2(d3)       # d2: [b, 64, 256, 256]
        d2 = torch.cat((x1, d2), dim=1)     # d2: [b, 128, 256, 256]
        d2 = self.Up_conv2(d2)      # d2: [b, 64, 256, 256]

        d1 = self.Conv_1x1(d2)      # d1: [b, 3, 256, 256]

        return d1
