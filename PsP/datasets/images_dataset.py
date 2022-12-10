from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
import os
from torch.utils.data import Dataset
from PIL import Image
import glob
import pandas as pd
from PsP.datasets.utils import getTransforms, RGBAnno2Mask
import pickle
from tqdm import tqdm

class ImagesDataset(Dataset):
	def __init__(self, source_root, target_root, opts, target_transform=None, source_transform=None):
		self.source_paths = sorted(data_utils.make_dataset(source_root))
		self.target_paths = sorted(data_utils.make_dataset(target_root))
		self.source_transform = source_transform
		self.target_transform = target_transform
		self.opts = opts

	def __len__(self):
		return len(self.source_paths)

	def __getitem__(self, index):
		from_path = self.source_paths[index]
		from_im = Image.open(from_path)
		from_im = from_im.convert('RGB') if self.opts.label_nc == 0 else from_im.convert('L')

		to_path = self.target_paths[index]
		to_im = Image.open(to_path).convert('RGB')
		if self.target_transform:
			to_im = self.target_transform(to_im)

		if self.source_transform:
			from_im = self.source_transform(from_im)
		else:
			from_im = to_im

		return from_im, to_im



class ADE20KDS(Dataset):
    def __init__(self, dataPath) -> None:
        self.imgAnnoPairs = self.__getImgAnnoPair(dataPath)
        self.mappingDict = {0: 0} # set label unknown(0)
        self.mappingDict = self.__getMappingDict(self.imgAnnoPairs)
        self.imgTransform = getTransforms(mode='img')
        self.annoTransform = getTransforms(mode='anno')

    def __len__(self):
        return len(self.imgAnnoPairs)

    def __getitem__(self, idx):
        img = self.imgAnnoPairs[idx][0]
        img = Image.open(img)
        img = img.convert('RGB') # by NVLab
        imgTensor = self.imgTransform(img)
        anno = self.imgAnnoPairs[idx][1]
        anno = Image.open(anno)
        anno, _ = RGBAnno2Mask(anno, self.mappingDict)
        annoTensor = self.annoTransform(anno)*255

        return imgTensor, annoTensor


    def __getMappingDict(self, imgAnnoPairs):
        if len(glob.glob("mappingfiles/Name2Idx.csv")) == 0:
            # 若mapping.csv 不存在則從ADE20K中的object.txt中建立mapping.csv
            os.makedirs("mappingfiles", exist_ok=True)
            
            for elem in tqdm(imgAnnoPairs):
                anno = Image.open(elem[1])
                _, self.mappingDict = RGBAnno2Mask(anno, self.mappingDict)

            df = pd.DataFrame(self.mappingDict.items())
            df.columns = ['OriginalIdx', 'NewIdx']
            df.to_csv('mappingfiles/Name2Idx.csv', encoding="UTF-8", index=False)


        print("mapping file loaded!")
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
