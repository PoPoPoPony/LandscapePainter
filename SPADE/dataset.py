import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import glob
import pandas as pd
from utils import getTransforms
import torch
import numpy as np
import pickle
from utils import RGBAnno2Mask
from tqdm import tqdm


class ADE20KDS(Dataset):
    def __init__(self, dataPath) -> None:
        self.imgAnnoPairs = self.__getImgAnnoPair(dataPath)
        self.mappingDict = {0: 0} # set label unknown(0)
        self.mappingDict = self.__getMappingDict(dataPath, self.imgAnnoPairs)
        
        

        # df = pd.read_csv(f"{dataPath}/objectInfo119.csv", encoding="UTF-8")
        # idxs = df['Idx'].to_list()
        # names = []
        # for name in df['Name'].to_list():
        #     if ';' in name:
        #         names.append(name.split(";")[0])
        #     else:
        #         names.append(name)
        # self.mappingClass = dict(zip(idxs, names))
        self.imgTransform = getTransforms(mode='img')
        self.annoTransform = getTransforms(mode='anno')
        # print(self.mappingClass)

    def __len__(self):
        return len(self.imgAnnoPairs)

    def __getitem__(self, idx):
        img = self.imgAnnoPairs[idx][0]
        img = Image.open(img)
        img = img.convert('RGB') # by NVLab
        imgTensor = self.imgTransform(img)

        # print image which is not RGB
        # if imgTensor.shape[0] != 3:
        #     print(self.imgPaths[idx])

        anno = self.imgAnnoPairs[idx][1]
        anno = Image.open(anno)
        # print(np.array(anno).shape)
        anno = RGBAnno2Mask(anno, self.mappingDict)

        # 不知道為啥convert完之後不會到0~1之間= =
        annoTensor = self.annoTransform(anno)


        # print(annoTensor)

        # exit(0)


        # if annoTensor.shape[0] != 1:
        #     print(self.imgPaths[idx])


        return imgTensor, annoTensor


    def __getMappingDict(self, dataPath, imgAnnoPairs):
        if len(glob.glob("mappingfiles/Name2Idx.csv")) == 0:
            # 若mapping.csv 不存在則從ADE20K中的object.txt中建立mapping.csv
            os.makedirs("mappingfiles", exist_ok=True)
            # with open(f"{dataPath}/objects.txt",encoding="ISO-8859-1") as f:
            #     data = f.readlines()

            # data = data[1:] # 去除column name
            # store = []
            
            # for i in range(len(data)):
            #     log = data[i].split("\t")
            #     label = log[0]
            #     if ',' in label:
            #         label = label.split(",")[0]
            #     originalIdx = log[1]
            #     newIdx = i+1

            #     store.append([label, originalIdx, newIdx])

            # store.insert(0 , ["unknown", 0, 0]) # ADE20K中0為unknown

            # df = pd.DataFrame(store)
            # df.columns = ['Label', 'OriginalIdx', 'NewIdx']
            # df.to_csv('mappingfiles/Name2Idx.csv', encoding="UTF-8", index=False)

            for elem in tqdm(imgAnnoPairs):
                anno = Image.open(elem[1])
                _, self.mappingDict = RGBAnno2Mask(anno, self.mappingDict)

            df = pd.DataFrame(self.mappingDict.items())
            df.columns = ['OriginalIdx', 'NewIdx']
            df.to_csv('mappingfiles/Name2Idx.csv', encoding="UTF-8", index=False)



        df = pd.read_csv('mappingfiles/Name2Idx.csv', encoding="UTF-8")
        mappingDict = dict(df.loc[:, 'OriginalIdx':'NewIdx'].to_dict('split')['data'])

        return mappingDict


    def __getImgAnnoPair(self, dataPath):
        """
        only return nature_landscape folder in ADE20k
        """
        
        pklPath = f"{dataPath}/index_ade20k.pkl"
        with open(pklPath, 'rb') as f:
            data = pickle.load(f)

        folders = []
        imgfilenames = []
        
        for i in range(len(data['folder'])):
            if "nature_landscape" in data['folder'][i] or "urban" in data['folder'][i] or "industrial" in data['folder'][i]:
                folders.append(data['folder'][i])
                imgfilenames.append(data['filename'][i])
                

        annofilenames = [x.replace(".jpg", "_seg.png") for x in imgfilenames]

        imgAnnoPairs = []
        for i in range(len(folders)):
            imgAnnoPairs.append([
                os.path.join(folders[i], imgfilenames[i]),
                os.path.join(folders[i], annofilenames[i])
            ])

        return imgAnnoPairs