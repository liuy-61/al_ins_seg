import torch.nn as nn
from model import implement
from model import UNet

class UNet_Pick(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_Pick, self).__init__()
        # self.n_channels = n_channels
        # self.n_classes = n_classes
        # self.inConv = implement.inConv(self.n_channels, 64)
        # self.down1 = implement.Down(64, 128)
        # self.down2 = implement.Down(128, 256)
        # self.down3 = implement.Down(256, 512)
        # self.down4 = implement.Down(512, 1024)
        # self.up1 = implement.Up(1024, 512)
        # self.up2 = implement.Up(512, 256)
        # self.up3 = implement.Up(256, 128)
        # self.up4 = implement.Up(128, 64)
        # self.outConv = implement.outConv(64, n_classes)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.Unet = UNet(n_classes=n_classes, n_channels=n_channels)

        self.QAM = implement.QAM()
        self.OCM = implement.OCM()


    def forward(self, x, y):
        # x1 = self.inConv(x)
        # x2 = self.down1(x1)
        # x3 = self.down2(x2)
        # x4 = self.down3(x3)
        # x5 = self.down4(x4)
        # x6 = self.up1(x5, x4)
        # x7 = self.up2(x6, x3)
        # x8 = self.up3(x7, x2)
        # x9 = self.up4(x8, x1)
        # res1 = self.outConv(x9)

        res1 = self.Unet(x)
        res2 = self.QAM(x, y)
        res2 = self.OCM(res2)

        return res1, res2

