import torch.nn as nn
import  torch
import torch.nn.functional as F
from model.architecture import VGG19

class GANLoss(nn.Module):
    def __init__(self, realLabel, fakeLabel) -> None:
        super(GANLoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.realLabelTensor = torch.FloatTensor(1).fill_(realLabel).to(self.device)
        self.realLabelTensor.requires_grad_(False)
        self.fakeLabelTensor = torch.FloatTensor(1).fill_(fakeLabel).to(self.device)
        self.fakeLabelTensor.requires_grad_(False)
        

    # def get_zero_tensor(self, input):
    #     if self.zero_tensor is None:
    #         self.zero_tensor = self.Tensor(1).fill_(0)
    #         self.zero_tensor.requires_grad_(False)
    #     return self.zero_tensor.expand_as(input)

    def getAdversarialLoss(self, inputTensor, target_is_real):
        if target_is_real:
            gtTensor = self.realLabelTensor.expand_as(inputTensor)
        else:
            gtTensor = self.fakeLabelTensor.expand_as(inputTensor)

        loss = F.binary_cross_entropy_with_logits(inputTensor, gtTensor)

        return loss


    def __call__(self, input_lst, target_is_real):
        loss = 0

        for pred in input_lst:
            pred = pred[-1] # adversarial loss 只看最後一層discriminator輸出的值
            lossTensor = self.getAdversarialLoss(pred, target_is_real)
            loss+=lossTensor

        return loss/len(input_lst)



def getFeatureMathcingLoss(pred_fake, pred_real):
    criterion = torch.nn.L1Loss()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    featureMatchingLoss = torch.FloatTensor(1).fill_(0).to(device)

    for i in range(len(pred_fake)): # discriminator 數量
        for j in range(len(pred_fake[i])-1): # features 數量，最後一層輸出由GAN loss評估
            loss = criterion(pred_fake[i][j], pred_real[i][j].detach())  # pred fake應該也要 detach()?
            featureMatchingLoss+= loss*10/len(pred_fake) # 10/discriminator 是官方預設的 feature matching loss 的 lambda


    return featureMatchingLoss




class VGGLoss(nn.Module):
    def __init__(self) -> None:
        super(VGGLoss, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.VGG = VGG19().to(device)
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]


    def forward(self, X, Y):
        vggX, vggY = self.VGG(X), self.VGG(Y)
        vggLoss = 0

        for i in range(len(vggX)):
            loss = self.criterion(vggX[i], vggY[i].detach()) # vggX 應該也要 detach()?
            vggLoss+=loss*self.weights[i]

        return vggLoss*10 # 10 是官方預設的 VGG loss 的 lambda