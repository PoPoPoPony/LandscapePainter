import sys

sys.path.append("..")
sys.path.append("../..")

from SPADE.model.generator import Generator
from PsP.models.psp import pSp
import torch
from SPADE.utils import getTransforms
import pandas as pd
# import os
# from PsP.options.test_options import TestOptions
from argparse import Namespace



def setupSPADE():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    annoTransform = getTransforms(mode='anno')
    g = Generator(styleSize=1399).to(device)
    g.load_state_dict(torch.load("../../SPADE_ckpt/epoche007.pt"))

    df = pd.read_csv('../../SPADE/mappingfiles/Name2Idx.csv', encoding="UTF-8")
    mappingDict = dict(df.loc[:, 'OriginalIdx':'NewIdx'].to_dict('split')['data'])

    obj = {
        "modelName": "SPADE",
        "model": g,
        "transform": annoTransform,
        "mappingDict": mappingDict
    }

    print("SPADE complete")

    return obj


def setupPsP():
    print("PsP start")
    annoTransform = getTransforms(mode='anno')
    # initial test parameters
    # test_opts = TestOptions().parse()
    test_opts = {}
    test_opts['checkpoint_path']="../../PsP_ckpt/iteration_200000.pt"
    test_opts['test_batch_size']=1
    test_opts['test_workers']=1
    test_opts['couple_outputs'] = False
    test_opts['resize_outputs'] = False
    test_opts['n_images'] = None
    test_opts['n_outputs_to_generate']=5
    test_opts['mix_alpha']=None
    test_opts['latent_mask']=None
    test_opts['resize_factors']=None

    print(123)
    ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
    print(456)
    opts = ckpt['opts']
    opts.update(test_opts) # 應該不用vars()
    if 'learn_in_w' not in opts:
        opts['learn_in_w'] = False
    if 'output_size' not in opts:
        opts['output_size'] = 1024
    opts = Namespace(**opts)

    net = pSp(opts) # load wights while initializing the pSp
    net.eval()
    net.cuda()

    df = pd.read_csv('../../SPADE/mappingfiles/Name2Idx.csv', encoding="UTF-8")
    mappingDict = dict(df.loc[:, 'OriginalIdx':'NewIdx'].to_dict('split')['data'])

    obj = {
        "modelName": "PsP",
        "model": net,
        "transform": annoTransform,
        "mappingDict": mappingDict
    }

    return obj
