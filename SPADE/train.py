from torchsummary import summary
from model.generator import Generator
from model.discriminator import MultiScaleDiscriminator
from dataset import ADE20KDS
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from model.loss import GANLoss, getFeatureMathcingLoss, VGGLoss
from writer import Writer
import glob
from tqdm import tqdm
from utils import convertAnnoTensor, concatImageAnno, devideFakeReal
from matplotlib import pyplot as plt

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


    # S = Generator(119).cuda()
    # summary(S, (119, 256, 256))
    # exit(0)


    # S = MultiScaleDiscriminator(119).cuda()
    # summary(S, (122, 256, 256))


    ds = ADE20KDS(dataPath="ADE20K_2021_17_01")
    trainLoader = DataLoader(ds, batch_size=1, shuffle=False)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    styleSize = len(ds.mappingDict)
    G = Generator(styleSize).to(device)
    G.apply(init_weights)
    D = MultiScaleDiscriminator(styleSize).to(device)
    D.apply(init_weights)
    lr_G = 0.0002
    lr_D = 0.0002
    beta1 = 0
    beta2 = 0.999

    G_opt = Adam(G.parameters(), lr=lr_G, betas=(beta1, beta2))
    D_opt = Adam(D.parameters(), lr=lr_D, betas=(beta1, beta2))

    # load checkPoint files
    G_pts = glob.glob("CheckPt/Generator/*.pt")
    D_pts = glob.glob("CheckPt/Discriminator/*.pt")

    if len(G_pts)>0:
        # sort epoch by filename
        G_pts.sort(key=lambda x:int(x.split('.')[-2][-3:]))
        
        G_pt = G_pts[-1]

        # d = torch.load(G_pt)
        # for i in d:
        #     print(i)
        # exit()


        G.load_state_dict(torch.load(G_pt))

        D_pt = D_pts[-1]
        D_pts.sort(key=lambda x:int(x.split('.')[-2][-3:]))
        D.load_state_dict(torch.load(D_pt))

        start_ep = int(G_pt.split('.')[-2][-3:])
    else:
        start_ep = 0


    EPOCHES = 50
    imgs = []
    G_losses = []
    D_losses = []
    criterionGAN = GANLoss(fakeLabel=0.0, realLabel=1.0)
    criterionVGG = VGGLoss()
    writer = Writer(rootPath='.')

    for epoch in range(start_ep, EPOCHES):
        print(f"Epoches : {epoch+1} / {EPOCHES}")
        G.train()
        D.train()
        for i, data in tqdm(enumerate(trainLoader)):
            img = data[0].to(device)
            anno = data[1].to(device)

            # a = anno[0].cpu().detach().numpy()
            # plt.imshow(a[0])
            # plt.show()

            # b = img[0].cpu().detach().numpy()
            # plt.imshow(b[0])
            # plt.show()
            # exit(0)


            print(styleSize)
            anno = convertAnnoTensor(anno, styleSize)

            G_opt.zero_grad()
            # sample latentVector from N(0, 1)
            # latentVector = torch.empty(256).normal_(0.0, 1.0).to(device) # initial by other method
            fakeImg = G(anno)
            fake_and_real = concatImageAnno(img, fakeImg, anno)
            pred_fake, pred_real = devideFakeReal(D(fake_and_real))

            loss_G = {}
            loss_G['GAN'] = criterionGAN(pred_fake, True)
            loss_G['Feature'] = getFeatureMathcingLoss(pred_fake, pred_real)
            loss_G['VGG'] = criterionVGG(fakeImg, img)

            loss_G_sum = sum(loss_G.values()).mean()
            loss_G_sum.backward()
            G_opt.step()
            
            D_opt.zero_grad()
            with torch.no_grad():
                fakeImg = G(anno).detach()
                fakeImg.requires_grad_()

            fake_and_real = concatImageAnno(img, fakeImg, anno)
            pred_fake, pred_real = devideFakeReal(D(fake_and_real))

            loss_D = {}
            loss_D['fake'] = criterionGAN(pred_fake, False)
            loss_D['real'] = criterionGAN(pred_real, True)
            loss_D_sum = sum(loss_D.values()).mean()

            loss_D_sum.backward()
            D_opt.step()

            G_losses.append({
                'GAN': loss_G['GAN'].item(),
                'Feature': loss_G['Feature'].item(),
                'VGG': loss_G['VGG'].item(),

            })
            D_losses.append({
                'fake': loss_D['fake'].item(),
                'real': loss_D['real'].item(),
            })


            if i%10==0 and i>5:
                print()
                print(G_losses[-1])
                print(D_losses[-1])

                with torch.no_grad():
                    writer.writeLoss("G", loss_G_sum.item())
                    writer.writeLoss("D", loss_D_sum.item())
                    # latentVector = torch.empty(256).normal_(0.0, 1.0).to(device) # initial by other method
                    fakeImg = G(anno).detach().to('cpu')
                    writer.writeResult(epoch, fakeImg, i)


        writer.writeCheckPt(epoch, G, "G")
        writer.writeCheckPt(epoch, D, "D")

        

        


