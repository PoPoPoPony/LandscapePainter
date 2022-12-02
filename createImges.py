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
filenames = glob.glob("OST300/*.png")


for i in range(len(filenames)):
    img = Image.open(filenames[i])
    filename = filenames[i].split("/")[-1][:-4]
    newFilename = f"stylegan2-pytorch/data/landscapedata/{filename}.jpg"

    # shutil.copyfile(filenames[i], newFilename)
    img.save(newFilename)