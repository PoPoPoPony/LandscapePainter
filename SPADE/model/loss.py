import torch.nn as nn
import  torch
import torch.nn.functional as F


class GANloss(nn.Module):
    def __init__(self, realLabel, fakeLabel, tensor=torch.FloatTensor) -> None:
        super(GANloss, self).__init__()

        self.realLabel = realLabel
        self.fakeLabel = fakeLabel
        self.zero_tensor = tensor
        self.Tensor = tensor

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)


    def getHingeLoss(self, input, target_is_real, for_discriminator):
        if for_discriminator:
            if target_is_real:
                minval = torch.min(input - 1, self.get_zero_tensor(input))
                loss = -torch.mean(minval)
            else:
                minval = torch.min(-input - 1, self.get_zero_tensor(input))
                loss = -torch.mean(minval)
        else:
            assert target_is_real, "The generator's hinge loss must be aiming for real"
            loss = -torch.mean(input)
        return loss

    
    def getAdversarialLoss(self, input, target_is_real):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if target_is_real:
            gtTensor = torch.FloatTensor(1).fill_(self.realLabel)
        else:
            gtTensor = torch.FloatTensor(1).fill_(self.fakeLabel)

        gtTensor.requires_grad_(False)
        gtTensor = gtTensor.expand_as(input).to(device)
        loss = F.binary_cross_entropy_with_logits(input, gtTensor)

        return loss


    def __call__(self, input, target_is_real, lossMode, for_discriminator=True, ):
        if lossMode=='hinge':
            return self.getHingeLoss(input, target_is_real, for_discriminator)
        elif lossMode=='ad':
            return self.getAdversarialLoss(input, target_is_real)
