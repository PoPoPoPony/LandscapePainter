import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch


class Discriminator(nn.Module):
    def __init__(self, styleSize) -> None:
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(3+styleSize, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(256, 512, 4, 2, 1)),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(512, 1, 4, 1)),
        )

    def forward(self, realImg, fakeImg, anno):
        fakeConcat = torch.cat([anno, fakeImg], dim=1)
        realConcat = torch.cat([anno, realImg], dim=1)

        fake_and_real = torch.cat([fakeConcat, realConcat], dim=0)
        print(fake_and_real.shape)
        pred = self.model(fake_and_real)

        fake = pred[:pred.size(0) // 2]
        real = pred[pred.size(0) // 2:]

        return fake, real
