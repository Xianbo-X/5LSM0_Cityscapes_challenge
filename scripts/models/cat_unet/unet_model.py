""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class cat_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(cat_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 64 // factor, bilinear)
        self.up2 = Up(512, 64 // factor, bilinear)
        self.up3 = Up(256, 64 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(320, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.up1(x5, x1)
        x7 = self.up2(x4, x1)
        x8 = self.up3(x3, x1)
        x9 = self.up4(x2, x1)
        x = torch.cat([x1,x9,x8,x7,x6],dim=1)
        logits = self.outc(x)
        return logits
