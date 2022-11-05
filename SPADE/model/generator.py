import torch.nn as nn
from model.architecture import SPADEResBlk
from torch.nn.utils import spectral_norm


class Generator(nn.Module):
    def __init__(self, ) -> None:
        super(Generator, self).__init__()

        self.l1 = nn.Linear(256, 16384) # 加入spectral_norm ???
        outputSizes = [
            # 1024,   # 暫時移除1024 ~ 1024
            1024,
            1024,
            1024,
            512,
            256,
            128,
            64,
        ]

        models = []
        for i in range(len(outputSizes)-1):
            models.append(SPADEResBlk(outputSizes[i], outputSizes[i+1]))
            models.append(nn.Upsample(scale_factor=2, mode='nearest'))

        self.models = nn.Sequential(*models)
        self.conv = spectral_norm(nn.Conv2d(64, 3, 3, 1, 1))
        self.tanh = nn.Tanh()


    def forward(self, x, s):
        x = self.l1(x)
        x = x.view(-1, 1024, 4, 4)

        for module in self.models:
            if type(module) == SPADEResBlk:
                x = module(x, s)
            else:
                x = module(x)

        x = self.conv(x)
        x = self.tanh(x)

        return x
