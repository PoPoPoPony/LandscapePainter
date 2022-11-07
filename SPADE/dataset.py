import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import glob
import pandas as pd
from utils import getTransforms, convertAnnoTensor
from torch.nn.functional import one_hot
import torch


class ADE20KDS(Dataset):
    def __init__(self, dataPath) -> None:
        self.imgPaths = glob.glob(f"{dataPath}/images/*/*.jpg")
        self.annoPaths = glob.glob(f"{dataPath}/annotations/*/*.png")
        df = pd.read_csv(f"{dataPath}/objectInfo150.csv", encoding="UTF-8")
        idxs = df['Idx'].to_list()
        names = []
        for name in df['Name'].to_list():
            if ';' in name:
                names.append(name.split(";")[0])
            else:
                names.append(name)
        self.mappingClass = dict(zip(idxs, names))
        self.imgTransform = getTransforms(mode='img')
        self.annoTransform = getTransforms(mode='anno')
        print(self.mappingClass)

    def __len__(self):
        return len(self.imgPaths)

    def __getitem__(self, idx):
        img = self.imgPaths[idx]
        img = Image.open(img)
        imgTensor = self.imgTransform(img)

        anno = self.annoPaths[idx]
        anno = Image.open(anno)
        annoTensor = self.annoTransform(anno)*255
        annoTensor = annoTensor.type(torch.int64)
        # annoTensor = torch.squeeze(annoTensor, 0)
        # annoTensor = annoTensor.view(-1)
        # print(annoTensor.shape)
        # annoTensor = one_hot(annoTensor, num_classes=151)
        # print(annoTensor.shape)

        # print(annoTensor)
        # exit(0)
        annoTensor = convertAnnoTensor(annoTensor)

        return imgTensor, annoTensor