from PIL import Image
import glob
from numpy import asarray
import numpy as np
from tqdm import tqdm
import pandas as pd
from copy import deepcopy
from torchvision import transforms


a = np.array([])
for annoPath in tqdm(glob.glob("ADE20K Outdoors/annotations/*/*.png")):
    # print(annoPath)
    img = Image.open(annoPath)
    # img2 = Image.open(f"ADE20K Outdoors/images/training/{annoPath.split('/')[-1][:-4]}.jpg")
    # c = transforms.ToTensor()
    # img2.save(f"{annoPath.split('/')[-1]}")
    # t = np.array(img)
    # print(t)

    a = np.append(a, asarray(img))
    a = np.unique(a)

df = pd.read_csv(f"ADE20K Outdoors/objectInfo150.csv", encoding="UTF-8")
idxs = df['Idx'].to_list()
names = []
for name in df['Name'].to_list():
    if ';' in name:
        names.append(name.split(";")[0])
    else:
        names.append(name)

notUseClasses = [x for x in range(151) if x not in a]
print(notUseClasses)
originalImgClasses = zip(idxs, names)
newImgClasses = deepcopy(originalImgClasses)
newImgClasses = [list(x) for x in newImgClasses if x[0] not in notUseClasses]
mappingDict = {}

for i in range(len(newImgClasses)):
    mappingDict[newImgClasses[i][0]] = i
    newImgClasses[i][0] = i

a = np.array([])
for annoPath in tqdm(glob.glob("ADE20K Outdoors/annotations/*/*.png")):
    img = Image.open(annoPath)
    pixels = img.load()
    for i in range(img.size[0]): # for every pixel:
        for j in range(img.size[1]):
            pixels[i, j] = mappingDict[pixels[i, j]]

    newPath = f"ADE20K Outdoors/annotations2/{annoPath.split('/')[-1]}"
    img.save(newPath)

    img2 = Image.open(newPath)
    a = np.append(a, asarray(img))
    a = np.unique(a)

print(a)
# print(newImgClasses)
# print(mappingDict)

# l = len(newImgClasses)
# newImgClasses = pd.DataFrame(newImgClasses)
# newImgClasses.columns = ["Idx", "Name"]
# newImgClasses.to_csv(f"ADE20K Outdoors/objectInfo{l}.csv", encoding="UTF-8", index=False)

