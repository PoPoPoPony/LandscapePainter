import shutil
import pickle
import os
from tqdm import tqdm
from io import BytesIO
import glob
from PIL import Image

# 處理ADE20K資料
# pklPath = f"SPADE/ADE20K_2021_17_01/index_ade20k.pkl"
# with open(pklPath, 'rb') as f:
#     data = pickle.load(f)

# folders = []
# imgfilenames = []

# for i in range(len(data['folder'])):
#     if "nature_landscape" in data['folder'][i] or "urban" in data['folder'][i] or "industrial" in data['folder'][i]:
#         folders.append(data['folder'][i])
#         imgfilenames.append(data['filename'][i])


# oldfilePath = [os.path.join("SPADE", folders[i], imgfilenames[i]) for i in range(len(folders))]
# newfilePath = [os.path.join(f"stylegan2-pytorch/data/landscapedata", imgfilenames[i]) for i in range(len(folders))]

# os.makedirs("stylegan2-pytorch/data/landscapedata", exist_ok=True)

# for i in tqdm(range(len(oldfilePath))):
#    shutil.copyfile(oldfilePath[i], newfilePath[i])



# 處理OST300、landscape、LHQ
# filenames = glob.glob("OST300/*.png")


# for i in range(len(filenames)):
#     img = Image.open(filenames[i])
#     filename = filenames[i].split("/")[-1][:-4]
#     newFilename = f"stylegan2-pytorch/data/landscapedata/{filename}.jpg"

#     # shutil.copyfile(filenames[i], newFilename)
#     img.save(newFilename)


import pandas as pd
import numpy as np
import glob
from PIL import Image
from tqdm import tqdm


# mapping psp encoder anno data(img -> img)，(使label class 連續)
# df = pd.read_csv(f"SPADE/ADE20K Outdoors/objectInfo119.csv", encoding="UTF-8")
# mapping_dict = dict(df.to_dict("split")['data'])
# mapping_dict = {y: x for x, y in mapping_dict.items()}

# folderpath = glob.glob(f"PsP/data/*/anno/*.png")

# func = np.vectorize(lambda x, *y:mapping_dict[x])

# for path in tqdm(folderpath):
#     img = Image.open(path)
#     npy = np.array(img)
#     npy = func(npy)

#     newImg = Image.fromarray(np.uint8(npy))
#     newImg.save(path)





# generate oldIdx and newIdx mapping(for ade20k outdoor)

# folderpath = glob.glob(f"SPADE/ADE20K Outdoors/annotations/training/*.png")
# a = np.array([])

# for path in tqdm(folderpath):
#     img = Image.open(path)
#     npy = np.asarray(img)
#     u = np.unique(npy)
#     a = np.append(a, u)


# a = np.unique(a)
# a = a.tolist()
# a = [int(x) for x in a]

# df = pd.DataFrame(enumerate(a))
# print(df)
# df.columns = ['newIdx', 'oldIdx']
# df.to_csv(f"SPADE/ADE20K Outdoors/objectInfo{len(a)}.csv", encoding="UTF-8", index=False)

# generate oldIdx and newIdx mapping(for ade20k outdoor)


# folderpath = glob.glob(f"PsP/data/*/anno/*.png")
# img = Image.open(folderpath[0])
# npy = np.array(img)
# print(npy.shape)
# print(np.unique(npy))

a = []

for _ in range(119):
    temp = np.random.choice(range(256), replace=True, size=3)
    temp = temp.tolist()
    temp = [int(x) for x in temp]
    a.append(temp)

print(a)