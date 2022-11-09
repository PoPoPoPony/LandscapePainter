import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from torchvision.transforms import InterpolationMode
# from model.generator import Generator
# from model.discriminator import Discriminator
import os

# for original imag
def getTransforms(mode):
    transformList = []

    newSize = (256, 256)
    if mode == 'img':
        transformList.append(transforms.Resize(newSize, interpolation=InterpolationMode.BICUBIC))
    elif mode == 'anno':
        transformList.append(transforms.Resize(newSize, interpolation=InterpolationMode.NEAREST))
        transformList.append(ToOneHot(151))

    transformList.append(transforms.ToTensor())
    return transforms.Compose(transformList)


def convertAnnoTensor(annoTensor):
    w = annoTensor.size(1)
    h = annoTensor.size(2)
    oneHotEncondingTensor = np.zeros((151, h, w), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            c = annoTensor[0, i, j]
            oneHotEncondingTensor[c, i, j]=1
        
    return oneHotEncondingTensor


class ToOneHot(object):
    """ Convert the input PIL image to a one-hot torch tensor """
    def __init__(self, n_classes=None):
        self.n_classes = n_classes

    def onehot_initialization(self, a):
        if self.n_classes is None:
            self.n_classes = len(np.unique(a))
        out = np.zeros(a.shape + (self.n_classes, ), dtype=int)
        out[self.__all_idx(a, axis=2)] = 1
        return out

    def __all_idx(self, idx, axis):
        grid = np.ogrid[tuple(map(slice, idx.shape))]
        grid.insert(axis, idx)
        return tuple(grid)

    def __call__(self, img):
        img = np.array(img)
        one_hot = self.onehot_initialization(img)
        return one_hot