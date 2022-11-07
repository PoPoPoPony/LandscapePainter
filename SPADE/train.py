from torchsummary import summary
from model.generator import Generator
from model.discriminator import Discriminator
from dataset import ADE20KDS
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from model.loss import GANloss
from utils import writeCheckPt
import glob


# def init_weights(m):
#     if type(m) is nn.Linear:
#         nn.init.xavier_normal_(m.weight)
#         m.bias.data.fill_(0.0)
#     elif type(m) is nn.Conv2d:
#         nn.init.xavier_normal_(m.weight)
#         m.bias.data.fill_(0.0)


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


if __name__ == '__main__':
    # S = SPADE(1024).cuda()
    # summary(S, [(1024, 4, 4), (1, 512, 512)])
    # S = SPADEResBlk(1024, 1024).cuda()
    # summary(S, [(1024, 4, 4), (1, 512, 512)])
    # S = Generator().cuda()
    # summary(S, [(1, 256), (1, 512, 512)])
    # S = Discriminator().cuda()
    # summary(S, [(3, 512, 512), (1, 512, 512)])

    ds = ADE20KDS(dataPath="ADE20K Outdoors")
    trainLoader = DataLoader(ds, batch_size=1, shuffle=False)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    G = Generator(151).cuda()
    G.apply(init_weights)
    D = Discriminator(151).cuda()
    D.apply(init_weights)
    lr_G = 0.0001
    lr_D = 0.0004
    beta1 = 0
    beta2 = 0.999

    G_opt = Adam(G.parameters(), lr=lr_G, betas=(beta1, beta2))
    D_opt = Adam(D.parameters(), lr=lr_D, betas=(beta1, beta2))

    # load checkPoint files
    G_pts = glob.glob("CheckPt/Generator/*.pt")
    D_pts = glob.glob("CheckPt/Discriminator/*.pt")

    if len(G_pts)>0:
        G_pt = G_pts[-1]
        G.load_state_dict(torch.load(G_pt))

        D_pt = D_pts[-1]
        D.load_state_dict(torch.load(D_pt))

        start_ep = int(G_pt.split('.')[-2][-3:])
    else:
        start_ep = 0

    EPOCHES = 1
    G.train()
    D.train()
    imgs = []
    G_losses = []
    D_losses = []
    criterion = GANloss(fakeLabel=0.0, realLabel=1.0)

    for epoch in range(start_ep, EPOCHES):
        for i, data in enumerate(trainLoader):
            G_opt.zero_grad()
            D_opt.zero_grad()

            print(f"Epoches : {epoch+1} / {EPOCHES}")
            img = data[0].to(device)
            anno = data[1].to(device)

            # sample latentVector with N(0, 1)
            latentVector = torch.empty(256).normal_(0.0, 1.0).to(device) # initial by other method
            fakeImg = G(latentVector, anno)
            
            pred_fake = D(fakeImg, anno)
            loss_D_fake = criterion(pred_fake, False, lossMode='ad')

            pred_real = D(img, anno)
            loss_D_true = criterion(pred_real, True, lossMode='ad')

            loss_G = criterion(pred_fake, True)
            loss_D = loss_D_fake + loss_D_true

            loss_G.backward()
            G_opt.step()

            loss_D.backward()
            D_opt.step()

            G_losses.append(loss_G.detach().cpu())
            D_losses.append(loss_D.detach().cpu())

        writeCheckPt("CheckPt", epoch, G, "G")
        writeCheckPt("CheckPt", epoch, D, "D")

        


