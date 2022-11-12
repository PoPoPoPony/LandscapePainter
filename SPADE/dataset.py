import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import glob
import pandas as pd
from utils import getTransforms
import torch
import numpy as np


class ADE20KDS(Dataset):
    def __init__(self, dataPath) -> None:
        self.imgPaths = glob.glob(f"{dataPath}/images/*/*.jpg")
        self.annoPaths = glob.glob(f"{dataPath}/annotations2/*.png")
        df = pd.read_csv(f"{dataPath}/objectInfo119.csv", encoding="UTF-8")
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
        img = img.convert('RGB') # by NVLab
        imgTensor = self.imgTransform(img)

        # print image which is not RGB
        if imgTensor.shape[0] != 3:
            print(self.imgPaths[idx])

        anno = self.annoPaths[idx]
        anno = Image.open(anno)

        annoTensor = self.annoTransform(anno)*255.0
        if annoTensor.shape[0] != 1:
            print(self.imgPaths[idx])


        return imgTensor, annoTensor