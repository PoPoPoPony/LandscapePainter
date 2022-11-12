import torch.nn as nn
from torch.nn.functional import interpolate
from torch.nn.utils import spectral_norm
from torchvision.models import vgg19

class SPADE(nn.Module):
    def __init__(self, featureSize, styleSize) -> None:
        super(SPADE, self).__init__()
        self.bn = nn.BatchNorm2d(featureSize) # affine=False
        self.convStyle = nn.Sequential(
            spectral_norm(nn.Conv2d(styleSize, 128, 3, 1, 1)),
            nn.ReLU()
        )

        self.convGamma = spectral_norm(nn.Conv2d(128, featureSize, 3, 1, 1))
        self.convBeta = spectral_norm(nn.Conv2d(128, featureSize, 3, 1, 1))

    def forward(self, x, s):
        x = self.bn(x)
        s = interpolate(s, size=(x.size(2), x.size(3)), mode='nearest')
        s = self.convStyle(s)
        gamma = self.convGamma(s)
        beta = self.convBeta(s)

        return x*(1+gamma)+beta


class SPADEResBlk(nn.Module):
    def __init__(self, featureSize, outputSize, styleSize) -> None:
        super(SPADEResBlk, self).__init__()

        # 直接輸出k 還是要 (in+k)/2 ???
        middleSize = min(featureSize, outputSize)

        self.block1 = nn.Sequential(
            SPADE(featureSize, styleSize), 
            nn.LeakyReLU(0.2, inplace=True), 
            spectral_norm(nn.Conv2d(featureSize, middleSize, 3, 1, 1)),
        )

        self.block2 = nn.Sequential(
            SPADE(middleSize, styleSize), 
            nn.LeakyReLU(0.2, inplace=True), 
            spectral_norm(nn.Conv2d(middleSize, outputSize, 3, 1, 1)),
        )

        self.blockSkip = nn.Sequential(
            SPADE(featureSize, styleSize), 
            nn.LeakyReLU(0.2, inplace=True), 
            spectral_norm(nn.Conv2d(featureSize, outputSize, 3, 1, 1, bias=False)),
        )

    def forward(self, x, s):
        skip = self.forwardBlock(self.blockSkip, x, s)

        y = self.forwardBlock(self.block1, x, s)
        y = self.forwardBlock(self.block2, y, s)
        

        return y+skip


    def forwardBlock(self, block, x, s):
        for module in block:
            if type(module) == SPADE:
                x = module(x, s)
            else:
                x = module(x)

        return x




class VGG19(nn.Module):
    def __init__(self) -> None:
        super(VGG19, self).__init__()
        vgg_pretrained_features = vgg19(pretrained=True).features

        self.block1 = nn.Sequential(*vgg_pretrained_features[:2])
        self.block2 = nn.Sequential(*vgg_pretrained_features[2:7])
        self.block3 = nn.Sequential(*vgg_pretrained_features[7:12])
        self.block4 = nn.Sequential(*vgg_pretrained_features[12:21])
        self.block5 = nn.Sequential(*vgg_pretrained_features[21:30])

        for param in self.parameters():
            param.requires_grad_(False)

    def forward(self, x):
        tensor1 = self.block1(x)
        tensor2 = self.block2(tensor1)
        tensor3 = self.block3(tensor2)
        tensor4 = self.block4(tensor3)
        tensor5 = self.block5(tensor4)

        return [tensor1, tensor2, tensor3, tensor4, tensor5]
