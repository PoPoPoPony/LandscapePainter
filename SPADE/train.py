from torchsummary import summary
from model.architecture import SPADE
from model.architecture import SPADEResBlk
from model.generator import Generator
from model.discriminator import Discriminator


if __name__ == '__main__':
    # S = SPADE(1024).cuda()
    # summary(S, [(1024, 4, 4), (1, 512, 512)])

    # S = SPADEResBlk(1024, 1024).cuda()
    # summary(S, [(1024, 4, 4), (1, 512, 512)])

    # S = Generator().cuda()
    # summary(S, [(1, 256), (1, 512, 512)])

    S = Discriminator().cuda()
    summary(S, [(3, 512, 512), (1, 512, 512)])

    pass