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


def writeCheckPt(filePath, epoch, model, modelType):
    if modelType == 'G':
        filePath = os.path.join(filePath, 'Generator')
    elif modelType == 'D':
        filePath = os.path.join(filePath, 'Discriminator')
    
    epoch = str(epoch).zfill(3)
    os.makedirs(filePath, exist_ok=True)
    fileName = f"epoche{epoch}.pt"
    model.save(model.state_dict(), fileName)
    