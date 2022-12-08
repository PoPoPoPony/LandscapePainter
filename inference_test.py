from PIL import Image
import glob
from SPADE.model.generator import Generator
import torch
from SPADE.utils import getTransforms, RGBAnno2Mask, convertAnnoTensor
import pandas as pd
import numpy as np
import sys


# sys.path.append("/SPADE/")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# filepaths = glob.glob("testfolder/SPADE*.png")
# annoTransform = getTransforms(mode='anno')
# g = Generator(styleSize=1399).to(device)
# g.load_state_dict(torch.load("testfolder/SPADE/epoche007.pt"))

# df = pd.read_csv('SPADE/mappingfiles/Name2Idx.csv', encoding="UTF-8")
# mappingDict = dict(df.loc[:, 'OriginalIdx':'NewIdx'].to_dict('split')['data'])


# for filepath in filepaths:
#     anno = Image.open(filepath)
#     anno, _ = RGBAnno2Mask(anno, mappingDict)
#     annoTensor = annoTransform(anno)*255
#     annoTensor = annoTensor.to(device)
#     annoTensor = torch.unsqueeze(annoTensor, 0)
#     annoTensor = convertAnnoTensor(annoTensor, 1399)
#     fakeImage = g(annoTensor).detach().to('cpu')
#     imageNpy = fakeImage.numpy()[0]
#     imageNpy = (np.transpose(imageNpy, (1, 2, 0))+1)/2.0*255.0
#     imageNpy = np.clip(imageNpy, 0, 255)
#     imageNpy = imageNpy.astype(np.uint8)

#     image = Image.fromarray(imageNpy)
#     image.show()

