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
from model.loss import GANloss, GANLoss2
from writer import Writer
import glob
from tqdm import tqdm


# def init_weights(m):
#     if type(m) is nn.Linear:
#         nn.init.xavier_normal_(m.weight)
#         m.bias.data.fill_(0.0)
#     elif type(m) is nn.Conv2d:
#         nn.init.xavier_normal_(m.weight)
#         m.bias.data.fill_(0.0)


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


if __name__ == '__main__':
    # S = SPADE(1024).cuda()
    # summary(S, [(1024, 4, 4), (1, 512, 512)])
    # S = SPADEResBlk(1024, 1024).cuda()
    # summary(S, [(1024, 4, 4), (1, 512, 512)])
    # S = Generator(151).cuda()
    # summary(S, [(1, 256), (151, 512, 512)])
    # S = Discriminator().cuda()
    # summary(S, [(3, 512, 512), (1, 512, 512)])

    ds = ADE20KDS(dataPath="ADE20K Outdoors")
    trainLoader = DataLoader(ds, batch_size=10, shuffle=False)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    G = Generator(151).to(device)
    G.apply(init_weights)
    D = Discriminator(151).to(device)
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

    EPOCHES = 50
    imgs = []
    G_losses = []
    D_losses = []
    criterion = GANloss(fakeLabel=0.0, realLabel=1.0)
    criterion2 = GANLoss2()
    writer = Writer(rootPath='.')
    imgList = []

    for epoch in range(start_ep, EPOCHES):
        print(f"Epoches : {epoch+1} / {EPOCHES}")
        G.train()
        D.train()
        for i, data in tqdm(enumerate(trainLoader)):
            img = data[0].to(device)
            anno = data[1].to(device)

            # for demo
            if i < 5:
                imgList.append(anno)

            G_opt.zero_grad()
            # sample latentVector from N(0, 1)
            latentVector = torch.empty(256).normal_(0.0, 1.0).to(device) # initial by other method
            fakeImg = G(latentVector, anno)
            pred_real, pred_fake = D(img, fakeImg, anno)
            pred_real2 = pred_real.clone().detach().requires_grad_(True)
            pred_fake2 = pred_fake.clone().detach().requires_grad_(True)

            # loss_G = criterion(pred_fake, True, lossMode='ad')
            loss_G = criterion2(pred_fake, True,)
            loss_G.backward(retain_graph=True)
            G_opt.step()

            D_opt.zero_grad()
            # loss_D_fake = criterion(pred_fake.detach(), False, lossMode='ad')
            loss_D_fake = criterion2(pred_fake2, False)

            # loss_D_true = criterion(pred_real, True, lossMode='ad')
            loss_D_true = criterion2(pred_real2, True)
            loss_D = loss_D_fake + loss_D_true

            loss_D.backward()
            D_opt.step()

            G_losses.append(loss_G.detach().to('cpu'))
            D_losses.append(loss_D.detach().to('cpu'))

            if i%10==0 and i>5:
                writer.writeLoss("G", loss_G.detach().to('cpu').item())
                writer.writeLoss("D", loss_D.detach().to('cpu').item())
                G.eval()
                with torch.no_grad():
                    for i in range(len(imgList[-5:])):
                        latentVector = torch.empty(256).normal_(0.0, 1.0).to(device) # initial by other method
                        fakeImg = G(latentVector, imgList[i]).detach().to('cpu')
                        writer.writeResult(epoch, fakeImg, i)


        writer.writeCheckPt(epoch, G, "G")
        writer.writeCheckPt(epoch, D, "D")

        

        


