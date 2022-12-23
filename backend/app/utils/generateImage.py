import sys

sys.path.append("..")
sys.path.append("../..")

import torch
from SPADE.utils import RGBAnno2Mask, convertAnnoTensor
import numpy as np
from PIL import Image

    
def generateImage(model_info, anno):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    modelName = model_info['modelName']
    model = model_info['model']
    annoTransform = model_info['transform']
    mappingDict = model_info['mappingDict']

    anno, _ = RGBAnno2Mask(anno, mappingDict)
    annoTensor = annoTransform(anno)*255
    annoTensor = annoTensor.to(device)
    annoTensor = torch.unsqueeze(annoTensor, 0)
    annoTensor = convertAnnoTensor(annoTensor, 1399)

    if modelName == 'PsP':
        annoTensor = annoTensor.float()
        fakeImage = model(annoTensor, randomize_noise=False, resize=False)
    elif modelName == 'SPADE':
        fakeImage = model(annoTensor)

    fakeImage = fakeImage.detach().to('cpu')
    imageNpy = fakeImage.numpy()[0]
    imageNpy = (np.transpose(imageNpy, (1, 2, 0))+1)/2.0*255.0
    imageNpy = np.clip(imageNpy, 0, 255)
    imageNpy = imageNpy.astype(np.uint8)

    image = Image.fromarray(imageNpy)
    return image


