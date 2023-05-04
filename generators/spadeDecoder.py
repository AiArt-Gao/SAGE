import torch
from torch import nn
import torch.nn.functional as F


class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out


class SPADEResnetBlock(nn.Module):
    def __init__(self, in_channel, parsing_channel):
        super().__init__()
        # Attributes

        # create conv layers
        self.conv_0 = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1)

        # define normalization layers
        self.norm_0 = SPADE(in_channel, parsing_channel)
        self.norm_1 = SPADE(in_channel, parsing_channel)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = x

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx

        return out

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


if __name__ == '__main__':
    device = torch.device('cpu')
    # spade = SPADE(256, 19)
    spade_block = SPADEResnetBlock(256, 15)
    seg = torch.randn((1, 19, 64, 64), device=device)
    fea = torch.randn((1, 256, 64, 64), device=device)
    out = spade_block(fea, seg)
    print(out.shape)