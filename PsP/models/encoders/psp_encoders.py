import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from models.encoders.helpers import get_blocks
from models.stylegan2.model import EqualLinear

class GradualStyleBlock(nn.Module):
    def __init__(self, in_channel, out_channel, spatial) -> None:
        super(GradualStyleBlock, self).__init__()

        self.out_channel = out_channel
        self.spatial = spatial
        num_pools = int(np.log2(spatial))

        modules = [
            nn.Conv2d(in_channel, out_channel, 3, 2, 1),
            nn.LeakyReLU()
        ]

        for _ in range(num_pools-1):
            modules.extend([
                nn.Conv2d(out_channel, out_channel, 3, 2, 1),
                nn.LeakyReLU()
            ])

        self.convs = nn.Sequential(*modules)
        self.linear = EqualLinear(out_channel, out_channel, lr_mul=1)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.out_channel)
        x = self.linear(x)

        return x



class GradualStyleEncoder(nn.Module):
    def __init__(self, style_size, w_plus_dim) -> None:
        """
        Args:
            sryle_size: segmentation map channel number
            w_plus_dim: w plus dimension in StyleGANv2
        """
        super(GradualStyleEncoder, self).__init__()
        num_layers = 50
        self.w_plus_dim = w_plus_dim

        

        self.input_layer = nn.Sequential(
            nn.Conv2d(style_size, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64)
        )

        self.body = nn.Sequential(*get_blocks())
        self.coarse_idx = 3
        self.middle_idx = 7
        style_layers = []

        for i in range(w_plus_dim):
            if i < self.coarse_idx:
                style_layer = GradualStyleBlock(512, 512, 16)
            elif i< self.middle_idx:
                style_layer = GradualStyleBlock(512, 512, 32)
            else:
                style_layer = GradualStyleBlock(512, 512, 64)
            style_layers.append(style_layer)

        self.style_layers = nn.Sequential(*style_layers)
        self.rechannel_layer1 = nn.Conv2d(256, 512, 1, 1, 0)
        self.rechannel_layer2 = nn.Conv2d(128, 512, 1, 1, 0)
        
    def forward(self, x):
        w_plus = []

        x = self.input_layer(x)
        modulelist = list(self.body._modules.values())

        for i, model in enumerate(modulelist):
            x = model(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x
        

        for i in range(self.coarse_idx):
            w_plus.append(self.style_layers[i](c3))

        _, _, H, W = c2.size()
        p3 = F.interpolate(c3, (H, W), mode='bilinear', align_corners=True)
        c2 = p3+self.rechannel_layer1(c2)

        for i in range(self.coarse_idx, self.middle_idx):
            w_plus.append(self.style_layers[i](c2))

        _, _, H, W = c1.size()
        p2 = F.interpolate(c2, (H, W), mode='bilinear', align_corners=True)
        c1 = p2+self.rechannel_layer2(c1)

        for i in range(self.middle_idx, self.w_plus_dim):
            w_plus.append(self.style_layers[i](c1))

        w_plus = torch.stack(w_plus, dim=1)
        return w_plus