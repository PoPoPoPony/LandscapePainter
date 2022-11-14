# from PIL import Image
# import glob
# from numpy import asarray
# import numpy as np
# from tqdm import tqdm
# import pandas as pd
# from copy import deepcopy
# from torchvision import transforms
# from PIL import Image

# a = np.array([])
# for annoPath in tqdm(glob.glob("ADE20K Outdoors/annotations/*/*.png")):
#     # print(annoPath)
#     img = Image.open(annoPath)
#     # img2 = Image.open(f"ADE20K Outdoors/images/training/{annoPath.split('/')[-1][:-4]}.jpg")
#     # c = transforms.ToTensor()
#     # img2.save(f"{annoPath.split('/')[-1]}")
#     # t = np.array(img)
#     # print(t)

#     a = np.append(a, asarray(img))
#     a = np.unique(a)

# print(a)
# # df = pd.read_csv(f"ADE20K Outdoors/objectInfo150.csv", encoding="UTF-8")
# # idxs = df['Idx'].to_list()
# # names = []
# # for name in df['Name'].to_list():
# #     if ';' in name:
# #         names.append(name.split(";")[0])
# #     else:
# #         names.append(name)

# # notUseClasses = [x for x in range(151) if x not in a]
# # print(notUseClasses)
# # originalImgClasses = zip(idxs, names)
# # newImgClasses = deepcopy(originalImgClasses)
# # newImgClasses = [list(x) for x in newImgClasses if x[0] not in notUseClasses]
# # mappingDict = {}

# # for i in range(len(newImgClasses)):
# #     mappingDict[newImgClasses[i][0]] = i
# #     newImgClasses[i][0] = i

# # a = np.array([])
# # for annoPath in tqdm(glob.glob("ADE20K Outdoors/annotations/*/*.png")):
# #     fileName = annoPath.split('\\')[-1]
# #     img = Image.open(annoPath)
# #     pixels = img.load()
# #     for i in range(img.size[0]): # for every pixel:
# #         for j in range(img.size[1]):
# #             pixels[i, j] = mappingDict[pixels[i, j]]

# #     newPath = f"ADE20K Outdoors/annotations2/{fileName}"
# #     img.save(newPath)

# #     img2 = Image.open(newPath)
# #     a = np.append(a, asarray(img))
# #     a = np.unique(a)

# # print(a)
# # print(newImgClasses)
# # print(mappingDict)



# a = np.array([])
# for annoPath in tqdm(glob.glob("ADE20K Outdoors/annotations/*/*.png")):
#     # print(annoPath)
#     img = Image.open(annoPath)
#     # img2 = Image.open(f"ADE20K Outdoors/images/training/{annoPath.split('/')[-1][:-4]}.jpg")
#     # c = transforms.ToTensor()
#     # img2.save(f"{annoPath.split('/')[-1]}")
#     # t = np.array(img)
#     # print(t)

#     a = np.append(a, asarray(img))
#     a = np.unique(a)

# print(a)
# # df = pd.read_csv(f"ADE20K Outdoors/objectInfo150.csv", encoding="UTF-8")
# # idxs = df['Idx'].to_list()
# # names = []
# # for name in df['Name'].to_list():
# #     if ';' in name:
# #         names.append(name.split(";")[0])
# #     else:
# #         names.append(name)

# # notUseClasses = [x for x in range(151) if x not in a]
# # print(notUseClasses)
# # originalImgClasses = zip(idxs, names)
# # newImgClasses = deepcopy(originalImgClasses)
# # newImgClasses = [list(x) for x in newImgClasses if x[0] not in notUseClasses]
# # mappingDict = {}

# # for i in range(len(newImgClasses)):
# #     mappingDict[newImgClasses[i][0]] = i
# #     newImgClasses[i][0] = i

# # a = np.array([])
# # for annoPath in tqdm(glob.glob("ADE20K Outdoors/annotations/*/*.png")):
# #     fileName = annoPath.split('\\')[-1]
# #     img = Image.open(annoPath)
# #     pixels = img.load()
# #     for i in range(img.size[0]): # for every pixel:
# #         for j in range(img.size[1]):
# #             pixels[i, j] = mappingDict[pixels[i, j]]

# #     newPath = f"ADE20K Outdoors/annotations2/{fileName}"
# #     img.save(newPath)

# #     img2 = Image.open(newPath)
# #     a = np.append(a, asarray(img))
# #     a = np.unique(a)

# # print(a)
# # print(newImgClasses)
# # print(mappingDict)

# # l = len(newImgClasses)
# # newImgClasses = pd.DataFrame(newImgClasses)
# # newImgClasses.columns = ["Idx", "Name"]
# # newImgClasses.to_csv(f"ADE20K Outdoors/objectInfo{l}.csv", encoding="UTF-8", index=False)


# from torchvision import transforms


# img = Image.open("ADE20K_2021_17_01/images/ADE/training/cultural/apse__indoor/ADE_train_00001472_seg.png")
# k = transforms.ToTensor()
# img = k(img)
# print(img.shape)
import pickle
from PIL import Image
import numpy as np

# with open("index_ade20k.pkl", 'rb') as f:
#     data = pickle.load(f)

# print(data.keys())
# print(data['wordnet_hypernym'])



# fileseg ="ADE_train_00006765_seg.png"

# with Image.open(fileseg) as io:
#     seg = np.array(io)

# # Obtain the segmentation mask, bult from the RGB channels of the _seg file
# R = seg[:,:,0]
# G = seg[:,:,1]
# B = seg[:,:,2]
# ObjectClassMasks = (R/10).astype(np.int32)*256+(G.astype(np.int32))

# print(ObjectClassMasks[:10, 1200:])
# print(ObjectClassMasks.shape)




"""



"""
import pandas as pd

# with open('objects.txt', 'rb') as f:
#     data = f.readlines()

# for i in range(len(data)):
#     data[i] = data[i].decode('')


# with open("objects.txt",encoding="ISO-8859-1") as f:
#     data = f.readlines()

# print(data[1].split("\t"))


# exit(0)


# with open('objects3.txt', 'w', encoding="UTF-8") as f:
#     f.writelines(data)



# store = []

# data = data[1:]
# for i in range(len(data)):
#     log = data[i].split("\t")
#     label = log[0]
#     if ',' in label:
#         label = label.split(",")[0]
#     originalIdx = log[1]
#     newIdx = i+1

#     store.append([label, originalIdx, newIdx])

# store.insert(0 , ["unknown", 0, 0])

# df = pd.DataFrame(store)
# df.columns = ['Label', 'OriginalIdx', 'NewIdx']
# df.to_csv('Name2Idx.csv', encoding="UTF-8", index=False)


# df = pd.read_csv('Name2Idx.csv', encoding="UTF-8")
# print(df.head())




# import pickle


with open('ADE20K_2021_17_01/index_ade20k.pkl', 'rb') as f:
    data = pickle.load(f)

print(data.keys())
print(data['filename']
[0])
print(data['folder'][0])
print(len(data['objectnames']))



# from dataset import ADE20KDS


# a = ADE20KDS('ADE20K_2021_17_01')
