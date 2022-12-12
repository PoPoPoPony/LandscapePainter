import sys

sys.path.append("LandscapePainter/")


import os
os.environ["PATH"] = os.environ["PATH"] + "/opt/conda/lib/python3.9/site-packages/ninja"
os.environ['CUDA_LAUNCH_BLOCKING']='1'
# os.system('cd ~/.cache/torch_extensions/fused')
# os.system('ninja')


from typing import Union, List
from app.utils.setup import setupSPADE, setupPsP
# from app.utils.setup import setupPsP
from app.utils.generateImage import generateImage
from app.utils.utils import img2Base64
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import torch 

print(torch.cuda.is_available())
# setup models
SPADE_info = setupSPADE()
PsP_info = setupPsP()

print("complete!")



app = FastAPI()

# CORS
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.post("/api/v1/GenerateSPADE")
def GenerateSPADE(anno: List[List[List[int]]] = None):
    anno = np.array(anno)
    anno = Image.fromarray(np.uint8(anno)).convert('RGB')
    img = generateImage(SPADE_info, anno)
    
    base64_str = img2Base64(img)

    return base64_str
    


@app.post("/api/v1/GeneratePsP")
def GeneratePsP(anno: List[List[List[int]]] = None):
    anno = np.array(anno)
    anno = Image.fromarray(np.uint8(anno)).convert('RGB')
    img = generateImage(PsP_info, anno)
    
    base64_str = img2Base64(img)

    return base64_str


