from PIL import Image
import glob
from numpy import asarray
import numpy as np
from tqdm import tqdm


a = np.array([])
for annoPath in tqdm(glob.glob("ADE20K Outdoors/annotations/*/*.png")):
    img = Image.open(annoPath)
    # img.show()

    a = np.append(a, asarray(img))
    a = np.unique(a)

print(a)

