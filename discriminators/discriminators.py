import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdapterBlock(nn.Module):
    def __init__(self, in_channel, output_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channel, output_channels, 1, padding=0),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, X):
        return self.model(X)


def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1 or classname.find('Conv2d') != -1:
        torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')


class AddCoords(nn.Module):
    """
    Source: https://github.com/mkocabas/CoordConv-pytorch/blob/master/CoordConv.py
    """

    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r
        self.device = torch.device('cuda')

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim, device=self.device).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim, device=self.device).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(
                torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5,
                                                                                 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret


class CoordConv(nn.Module):
    """
    Source: https://github.com/mkocabas/CoordConv-pytorch/blob/master/CoordConv.py
    """

    def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
        super().__init__()
        self.add_coords = AddCoords(with_r=with_r)
        in_size = in_channels + 2
        if with_r:
            in_size += 1
        self.conv = nn.Conv2d(in_size, out_channels, **kwargs)

    def forward(self, x):
        ret = self.add_coords(x)
        ret = self.conv(ret)
        return ret


class ResidualCCBlock(nn.Module):
    def __init__(self, in_planes, planes, kernel_size=3):
        super().__init__()
        p = kernel_size // 2
        self.network = nn.Sequential(
            CoordConv(in_planes, planes, kernel_size=kernel_size, padding=p),
            nn.LeakyReLU(0.2, inplace=True),
            CoordConv(planes, planes, kernel_size=kernel_size, stride=2, padding=p),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.network.apply(kaiming_leaky_init)

        self.proj = nn.Conv2d(in_planes, planes, 1, stride=2)

    def forward(self, x):
        y = self.network(x)

        identity = self.proj(x)

        y = (y + identity) / math.sqrt(2)
        return y


# semantic feature and RGB photo use same discriminator
class RGBDiscriminator(nn.Module):
    def __init__(self, in_channel=3):
        super().__init__()
        self.epoch = 0
        self.step = 0
        self.layers = nn.ModuleList(
            [
                ResidualCCBlock(32, 64),  # 6 256x256 -> 128x128
                ResidualCCBlock(64, 128),  # 5 128x128 -> 64x64
                ResidualCCBlock(128, 256),  # 4 64x64 -> 32x32
                ResidualCCBlock(256, 400),  # 3 32x32 -> 16x16
                ResidualCCBlock(400, 400),  # 2 16x16 -> 8x8
                ResidualCCBlock(400, 400),  # 1 8x8 -> 4x4
                ResidualCCBlock(400, 400),  # 7 4x4 -> 2x2
            ])

        self.fromRGB = nn.ModuleList(
            [
                AdapterBlock(in_channel, 32),
                AdapterBlock(in_channel, 64),
                AdapterBlock(in_channel, 128),
                AdapterBlock(in_channel, 256),
                AdapterBlock(in_channel, 400),
                AdapterBlock(in_channel, 400),
                AdapterBlock(in_channel, 400),
                AdapterBlock(in_channel, 400)
            ])
        self.final_layer = nn.Conv2d(400, 1 + 2, 2)
        self.img_size_to_layer = {2: 7, 4: 6, 8: 5, 16: 4, 32: 3, 64: 2, 128: 1, 256: 0}

    def forward(self, x_in, alpha, **kwargs):
        start = self.img_size_to_layer[x_in.shape[-1]]
        x = self.fromRGB[start](x_in)

        if kwargs.get('instance_noise', 0) > 0:
            x = x + torch.randn_like(x) * kwargs['instance_noise']

        for i, layer in enumerate(self.layers[start:]):
            if i == 1 and alpha < 1:
                x = alpha * x + (1 - alpha) * self.fromRGB[start + 1](
                    F.interpolate(x_in, scale_factor=0.5, mode='nearest'))

            x = layer(x)

        x = self.final_layer(x).reshape(x.shape[0], -1)

        prediction = x[..., 0:1]
        position = x[..., 1:]

        return prediction, position
