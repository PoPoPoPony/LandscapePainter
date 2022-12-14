import torch.nn as nn
from SPADE.model.architecture import SPADEResBlk
# from torch.nn.utils import spectral_norm
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, styleSize) -> None:
        super(Generator, self).__init__()

        self.conv1 = nn.Conv2d(styleSize, 1024, 3, padding=1)

        outputSizes = [
            1024,
            1024,
            1024,
            1024,
            512,
            256,
            128,
        ]

        models = []
        for i in range(len(outputSizes)-1):
            models.append(SPADEResBlk(outputSizes[i], outputSizes[i+1], styleSize))
            models.append(nn.Upsample(scale_factor=2, mode='nearest'))

        self.models = nn.Sequential(*models)
        self.lastSPADEResBlk = SPADEResBlk(128, 64, styleSize) # 因為最後一個沒有要upSample所以另外寫
        self.leaky = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(64, 3, 3, padding=1) # 官方這邊沒加spectral_norm
        self.tanh = nn.Tanh()
        self.newAnnoW, self.newAnnoH = self.computeLatentVectorSize(6)

    def forward(self, input1):
        s = input1

        x = F.interpolate(s, size=(self.newAnnoH, self.newAnnoW))
        x = self.conv1(x)

        

        for module in self.models:
            if type(module) == SPADEResBlk:
                x = module(x, s)
            else:
                x = module(x)

        
        x = self.lastSPADEResBlk(x, s)
        x = self.leaky(x)
        x = self.conv2(x)

        return x


    # 下採樣annotation map當作latentVector(並非隨機產生)
    def computeLatentVectorSize(self, upSampleTimes):
        # 圖都先用256*256
        annoW = 256 // (2**upSampleTimes)
        annoH = 256 // (2**upSampleTimes)

        return annoW, annoH

