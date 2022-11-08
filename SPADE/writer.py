import os
import torch
from PIL import Image
from torchvision import transforms
import glob
import numpy as np


class Writer():
    def __init__(self, rootPath) -> None:
        self.checkPtFilePathG = f"{rootPath}/CheckPt/Generator"
        self.checkPtFilePathD = f"{rootPath}/CheckPt/Discriminator"
        self.imagePath = f"{rootPath}/images"
        self.lossPath = f"{rootPath}/loss"

        os.makedirs(self.checkPtFilePathG, exist_ok=True)
        os.makedirs(self.checkPtFilePathD, exist_ok=True)
        os.makedirs(self.imagePath, exist_ok=True)
        os.makedirs(self.lossPath, exist_ok=True)


    def writeCheckPt(self, epoch, model, modelType):
        epoch = str(epoch).zfill(3)
        if modelType == 'G':
            fileName = os.path.join(self.checkPtFilePathG, f"epoche{epoch}.pt") 
        elif modelType == 'D':
            fileName = os.path.join(self.checkPtFilePathD, f"epoche{epoch}.pt") 
        torch.save(model.state_dict(), fileName)
        print(f"[INFO] Save Epoch {epoch} check point files")


    def writeResult(self, epoch, imageTensor, idx):
        imageTensor = imageTensor[0]
        folderPath = f"{self.imagePath}/epoch{epoch}"
        os.makedirs(folderPath, exist_ok=True)
        fileName = f"{folderPath}/{str(idx).zfill(3)}.jpg"
        image = transforms.ToPILImage()(imageTensor).convert('RGB')
        image.save(fileName)

    def writeLoss(self, modelType, loss):
        if modelType == 'G':
            filePath = f"{self.lossPath}/Generator.npy"
        elif modelType == 'D':
            filePath = f"{self.lossPath}/Discriminator.npy"

        if len(glob.glob(filePath))>0:
            data = np.load(filePath)
        else:
            data = np.array([])
        data = np.append(data, loss)
        np.save(filePath, data)
