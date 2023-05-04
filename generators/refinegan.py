import torch
import torch.nn as nn
import numpy as np
from kornia.filters import filter2d

from generators.custom_layers import EqualizedConv2d, EqualizedLinear, Upscale2d
from generators.UNet import U_Net


class AdaIN(nn.Module):
    def __init__(self, dimIn, dimOut, epsilon=1e-8):
        super(AdaIN, self).__init__()
        self.epsilon = epsilon
        self.styleModulator = nn.Linear(dimIn, 2*dimOut)
        self.dimOut = dimOut

        with torch.no_grad():
            self.styleModulator.weight *= 0.25
            self.styleModulator.bias.data.fill_(0)
        
    def forward(self, x, y):
        # x: N x C x W x H
        batchSize, nChannel, width, height = x.size()
        tmpX = x.view(batchSize, nChannel, -1)
        mux = tmpX.mean(dim=2).view(batchSize, nChannel, 1, 1)
        var_x = torch.clamp((tmpX*tmpX).mean(dim=2).view(batchSize, nChannel, 1, 1) - mux*mux, min=0)
        var_x = torch.rsqrt(var_x + self.epsilon)
        x = (x - mux) * var_x

        # Adapt style
        styleY = self.styleModulator(y)
        yA = styleY[:, : self.dimOut].view(batchSize, self.dimOut, 1, 1)
        yB = styleY[:, self.dimOut:].view(batchSize, self.dimOut, 1, 1)

        return yA * x + yB


class NoiseMultiplier(nn.Module):
    def __init__(self):
        super(NoiseMultiplier, self).__init__()
        self.module = nn.Conv2d(1, 1, 1, bias=False)
        self.module.weight.data.fill_(0)

    def forward(self, x):

        return self.module(x)


class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)

    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f[None, :, None]
        return filter2d(x, f, normalized=True)


class ToRGB(nn.Module):
    def __init__(self, in_channel, actvn=False):
        super().__init__()
        self.conv = EqualizedConv2d(in_channel,
                                    3,
                                    1,
                                    equalized=True,
                                    initBiasToZero=True)
        self.actvn = actvn

    def forward(self, x, skip=None):
        out = self.conv(x)

        if self.actvn:
            out = torch.sigmoid(out)

        return out


class ToParsing(nn.Module):
    def __init__(self, in_channel, actvn=False):
        super().__init__()
        self.conv = EqualizedConv2d(in_channel,
                                    19,
                                    1,
                                    equalized=True,
                                    initBiasToZero=True)
        # self.conv = nn.Conv2d(in_channel, 15, 3, 1, 1)
        # self.linear = nn.Linear()
        self.actvn = actvn

    def forward(self, x, skip=None):
        out = self.conv(x)

        if self.actvn:
            out = torch.sigmoid(out)

        return out


class MappingLayer(nn.Module):
    def __init__(self, dimIn, dimLatent, nLayers, leakyReluLeak=0.2):
        super(MappingLayer, self).__init__()
        self.FC = nn.ModuleList()

        inDim = dimIn
        for i in range(nLayers):
            self.FC.append(EqualizedLinear(inDim, dimLatent, lrMul=0.01, equalized=True, initBiasToZero=True))
            inDim = dimLatent

        self.activation = torch.nn.LeakyReLU(leakyReluLeak)

    def forward(self, x):
        for layer in self.FC:
            x = self.activation(layer(x))

        return x


class GNet(nn.Module):
    def __init__(self, dimInput=256, dimHidden=256, dimMapping=256, leakyReluLeak=0.2):
        super(GNet, self).__init__()
        self.epoch = 0
        self.step = 0
        self.device = None
        
        self.dimMapping = dimMapping
        self.scaleLayers = nn.ModuleList()

        self.adain00 = AdaIN(dimMapping, dimInput)
        self.adain01 = AdaIN(dimMapping, dimHidden)

        self.UNet = U_Net(input_ch=3)

        self.conv0 = EqualizedConv2d(dimInput, dimHidden, 3, equalized=True,
                                     initBiasToZero=True, padding=1)

        self.activation = torch.nn.LeakyReLU(leakyReluLeak)

        self.depthScales = [dimHidden]
        
        self.toRGBLayers = nn.ModuleList()

        self.toRGBLayers.append(ToRGB(dimHidden, actvn=False))

        self.blur = Blur()
        self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), 
                self.blur)

        self.depth_list = [256, 128, 64]

        for depth in self.depth_list:
            self.addScale(depth)
        

    def set_device(self, device):
        self.device = device
        
    def addScale(self, dimNewScale):

        lastDim = self.depthScales[-1]
        self.scaleLayers.append(nn.ModuleList())
        self.scaleLayers[-1].append(EqualizedConv2d(lastDim,
                                                    dimNewScale,
                                                    3,
                                                    padding=1,
                                                    equalized=True,
                                                    initBiasToZero=True))

        self.scaleLayers[-1].append(AdaIN(self.dimMapping, dimNewScale))
        self.scaleLayers[-1].append(EqualizedConv2d(dimNewScale,
                                                    dimNewScale,
                                                    3,
                                                    padding=1,
                                                    equalized=True,
                                                    initBiasToZero=True))
        self.scaleLayers[-1].append(AdaIN(self.dimMapping, dimNewScale))

        self.toRGBLayers.append(ToRGB(dimNewScale, actvn=False))
        self.depthScales.append(dimNewScale)

    def forward(self, mapping, feature, img_size, output_size, alpha, parsing):

        feature = self.activation(feature)
        feature = self.adain00(feature, mapping)
        feature = self.conv0(feature)  # [batchSize, 256, 64, 64) -> # [batchSize, 256, 64, 64)
        feature = self.activation(feature)        
        feature = self.adain01(feature, mapping)
        num_depth_scale = int(np.log2(output_size)) - int(np.log2(img_size))
        for nLayer, group in enumerate(self.scaleLayers):
            if nLayer + 1 > num_depth_scale:
                break
            skip = self.toRGBLayers[nLayer](feature)    # calculate the last second feature
            feature = Upscale2d(feature)            # Upsample (nearest neighborhood instead of biniliear)
            feature = group[0](feature)
            feature = self.activation(feature)      # activation function
            feature = group[1](feature, mapping)    # adaptive instance normalization
            feature = group[2](feature)
            feature = self.activation(feature)      # activation function
            feature = group[3](feature, mapping)    # adaptive instance normalization

        if num_depth_scale == 0:
            rgb = self.toRGBLayers[num_depth_scale](feature) 
        else:
            rgb = (1 - alpha) * self.upsample(skip) + alpha * self.toRGBLayers[num_depth_scale](feature)

        # in the first training stage, we do not train u-net, please unpack the comment in the second training stage
        # if num_depth_scale == 2:
        #     rgb = self.UNet(rgb, parsing)

        return rgb


class GNet_parsing(nn.Module):
    def __init__(self, dimInput=256, dimHidden=256, dimMapping=256, leakyReluLeak=0.2):
        super(GNet_parsing, self).__init__()
        self.epoch = 0
        self.step = 0
        self.device = None

        self.dimMapping = dimMapping
        self.scaleLayers = nn.ModuleList()

        self.adain00 = AdaIN(dimMapping, dimInput)
        self.adain01 = AdaIN(dimMapping, dimHidden)

        self.conv0 = EqualizedConv2d(dimInput, dimHidden, 3, equalized=True,
                                     initBiasToZero=True, padding=1)

        self.activation = torch.nn.LeakyReLU(leakyReluLeak)

        self.depthScales = [dimHidden]

        self.toRGBLayers = nn.ModuleList()

        self.toRGBLayers.append(ToParsing(dimHidden, actvn=False))

        self.blur = Blur()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            self.blur)

        self.depth_list = [256, 128, 64]

        for depth in self.depth_list:
            self.addScale(depth)

    def set_device(self, device):
        self.device = device

    def addScale(self, dimNewScale):

        lastDim = self.depthScales[-1]
        self.scaleLayers.append(nn.ModuleList())
        self.scaleLayers[-1].append(EqualizedConv2d(lastDim,
                                                    dimNewScale,
                                                    3,
                                                    padding=1,
                                                    equalized=True,
                                                    initBiasToZero=True))

        self.scaleLayers[-1].append(AdaIN(self.dimMapping, dimNewScale))
        self.scaleLayers[-1].append(EqualizedConv2d(dimNewScale,
                                                    dimNewScale,
                                                    3,
                                                    padding=1,
                                                    equalized=True,
                                                    initBiasToZero=True))
        self.scaleLayers[-1].append(AdaIN(self.dimMapping, dimNewScale))

        self.toRGBLayers.append(ToParsing(dimNewScale, actvn=False))
        self.depthScales.append(dimNewScale)

    def forward(self, mapping, feature, img_size, output_size, alpha):

        feature = self.activation(feature)
        feature = self.adain00(feature, mapping)
        feature = self.conv0(feature)  # [batchSize, 256, 64, 64) -> # [batchSize, 256, 64, 64)
        feature = self.activation(feature)
        feature = self.adain01(feature, mapping)
        num_depth_scale = int(np.log2(output_size)) - int(np.log2(img_size))
        for nLayer, group in enumerate(self.scaleLayers):
            if nLayer + 1 > num_depth_scale:
                break
            skip = self.toRGBLayers[nLayer](feature)  # calculate the last second feature
            feature = Upscale2d(feature)  # Upsample (nearest neighborhood instead of biniliear)
            feature = group[0](feature)
            feature = self.activation(feature)  # activation function
            feature = group[1](feature, mapping)  # adaptive instance normalization
            feature = group[2](feature)
            feature = self.activation(feature)  # activation function
            feature = group[3](feature, mapping)  # adaptive instance normalization

        if num_depth_scale == 0:
            rgb = self.toRGBLayers[num_depth_scale](feature)
        else:
            rgb = (1 - alpha) * self.upsample(skip) + alpha * self.toRGBLayers[num_depth_scale](feature)

        return rgb


if __name__ == '__main__':
    device = torch.device('cpu')
    to = ToParsing(256).to(device)
    fea = torch.randn((1, 256, 64, 64), device=device)
    out = to(fea)
    print(out.shape)