import torch.nn as nn
from torch.nn.utils import spectral_norm


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, styleSize) -> None:
        super(MultiScaleDiscriminator, self).__init__()

        self.dis1 = Discriminator(styleSize)
        self.dis2 = Discriminator(styleSize)

        self.avg1 = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.avg2 = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)


    def forward(self, x):
        results = []

        result = self.dis1(x)
        results.append(result)

        x = self.avg1(x)

        result = self.dis2(x)
        results.append(result)

        x = self.avg2(x)

        # # print(len(results))
        # # print(len(results[0]))
        # # print(results[0][0].shape)
        # exit(0)

        return results


class Discriminator(nn.Module):
    def __init__(self, styleSize) -> None:
        super(Discriminator, self).__init__()

        self.seq1 = nn.Sequential(
            spectral_norm(nn.Conv2d(3+styleSize, 64, 4, stride=2, padding=2)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.seq2 = nn.Sequential(
            spectral_norm(nn.Conv2d(64, 128, 4, stride=2, padding=2)),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.seq3 = nn.Sequential(
            spectral_norm(nn.Conv2d(128, 256, 4, stride=2, padding=2)),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
            
        self.seq4 = nn.Sequential(
            spectral_norm(nn.Conv2d(256, 512, 4, stride=1, padding=2)),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
            
        self.lastConv = nn.Conv2d(512, 1, 4, stride=1, padding=2)


    def forward(self, x):
        results = [x]
        
        x = self.seq1(x)
        results.append(x)
        x = self.seq2(x)
        results.append(x)
        x = self.seq3(x)
        results.append(x)
        x = self.seq4(x)
        results.append(x)
        x = self.lastConv(x)
        results.append(x)

        return results[1:] # input 不需要 return