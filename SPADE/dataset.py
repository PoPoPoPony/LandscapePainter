import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import glob
import pandas as pd
from utils import getTransforms, convertAnnoTensor, ToOneHot
from torch.nn.functional import one_hot
import torch
import numpy as np


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

        # print image which is not RGB
        if imgTensor.shape[0] != 3:
            print(self.imgPaths[idx])

        anno = self.annoPaths[idx]
        anno = Image.open(anno)
        # if idx < 5:
        #     anno.save(f"test{idx}.jpg")
        

        # print(anno)
        annoTensor = self.annoTransform(anno)
        annoTensor = annoTensor.type(torch.FloatTensor)
        if annoTensor.shape[0] != 151:
            print(self.imgPaths[idx])

        # print(annoTensor.shape)
        # print(annoTensor)
        # print(annoTensor.shape)
        # annoTensor = annoTensor.type(torch.int64)
        # annoTensor = torch.squeeze(annoTensor, 0)


        # print(annoTensor)
        # exit(0)
        # annoTensor = convertAnnoTensor(annoTensor)

        return imgTensor, annoTensor