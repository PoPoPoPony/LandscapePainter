import torch.nn as nn
from torch.nn.functional import interpolate
from torch.nn.utils import spectral_norm


class SPADE(nn.Module):
    def __init__(self, featureSize, styleSize) -> None:
        super(SPADE, self).__init__()
        self.bn = nn.BatchNorm2d(featureSize) # affine=False
        self.convStyle = nn.Sequential(
            spectral_norm(nn.Conv2d(styleSize, 128, 3, 1, 1)),
            nn.ReLU(inplace=True)
        )

        self.convGamma = spectral_norm(nn.Conv2d(128, featureSize, 3, 1, 1))
        self.convBeta = spectral_norm(nn.Conv2d(128, featureSize, 3, 1, 1))

    def forward(self, x, s):
        x = self.bn(x)
        s = interpolate(s, size=(x.size(2), x.size(3)), mode='nearest')
        s = self.convStyle(s)
        gamma = self.convGamma(s)
        beta = self.convBeta(s)

        return x*gamma+beta # return x(1+gamma)+beta ???


class SPADEResBlk(nn.Module):
    def __init__(self, featureSize, k, styleSize) -> None:
        super(SPADEResBlk, self).__init__()

        # 直接輸出k 還是要 (in+k)/2 ???
        self.block1 = nn.Sequential(
            SPADE(featureSize, styleSize), 
            nn.ReLU(inplace=True), 
            spectral_norm(nn.Conv2d(featureSize, k, 3, 1, 1)),
        )

        self.block2 = nn.Sequential(
            SPADE(k, styleSize), 
            nn.ReLU(inplace=True), 
            spectral_norm(nn.Conv2d(k, k, 3, 1, 1)),
        )

        self.blockSkip = nn.Sequential(
            SPADE(featureSize, styleSize), 
            nn.ReLU(inplace=True), 
            spectral_norm(nn.Conv2d(featureSize, k, 3, 1, 1)),
        )

    def forward(self, x, s):
        y1 = self.forwardBlock(self.block1, x, s)
        y2 = self.forwardBlock(self.block2, y1, s)
        skip = self.forwardBlock(self.blockSkip, x, s)

        return y2+skip


    def forwardBlock(self, block, x, s):
        for module in block:
            if type(module) == SPADE:
                x = module(x, s)
            else:
                x = module(x)

        return x
