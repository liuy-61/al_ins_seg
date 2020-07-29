import torch.nn as nn
from model import implement
from model import UNet

class UNet_Pick_cbam(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_Pick_cbam, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.Unet = UNet(n_classes=n_classes, n_channels=n_channels)

        self.QAM = implement.QAM_cbam()
        self.OCM = implement.OCM()


    def forward(self, x, y):

        res1 = self.Unet(x)
        res2 = self.QAM(x, y)
        res2 = self.OCM(res2)

        return res1, res2

