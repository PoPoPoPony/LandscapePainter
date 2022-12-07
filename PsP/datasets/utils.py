import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from torchvision.transforms import InterpolationMode
import torch


def getTransforms(mode):
    transformList = []
    newSize = (256, 256)

    if mode == 'img':
        transformList.append(transforms.Resize(newSize, interpolation=InterpolationMode.BICUBIC))
        transformList.append(transforms.ToTensor())
        transformList.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    elif mode == 'anno':
        transformList.append(transforms.Resize(newSize, interpolation=InterpolationMode.NEAREST))
        transformList.append(transforms.ToTensor())

    return transforms.Compose(transformList)


def convertAnnoTensor(annoTensor:torch.Tensor, styleSize:int) -> torch.Tensor:
    """
    convert annoTensor from label encoding to one-hot encoding

    Args:
        annoTensor: segmentation map
        styleSize: number of classes
    
    Returns:
        oneHotEncondingTensor
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    annoTensor = annoTensor.long()
    batchSize, _, h, w = annoTensor.size()
    oneHotEncondingTensor = torch.FloatTensor(batchSize, styleSize, h, w).zero_().to(device)
    oneHotEncondingTensor.scatter_(1, annoTensor, 1.0)
        
    return oneHotEncondingTensor


def RGBAnno2Mask(anno, mappingDict:dict):
    seg = np.array(anno)

    R = seg[:,:,0]
    G = seg[:,:,1]
    B = seg[:,:,2]
    ObjectClassMasks = (R/10).astype(np.int32)*256+(G.astype(np.int32))
    uniqueClasses = np.unique(ObjectClassMasks)

    maxIdx = max(mappingDict.values())
    for c in uniqueClasses:
        if c not in mappingDict:
            maxIdx+=1
            mappingDict[c] = maxIdx

    func = np.vectorize(lambda x, *y:mappingDict[x])
    ObjectClassMasks = func(ObjectClassMasks)

    return Image.fromarray(np.uint8(ObjectClassMasks)), mappingDict