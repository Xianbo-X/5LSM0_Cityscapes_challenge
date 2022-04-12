import torch
import torch.nn as nn
import torch.nn.functional as F
from scripts.unet.unet_parts import DoubleConv

class PositionalEncoding(nn.Module):
    def __init__(self,in_channels,in_height,in_width) -> None:
        """
        batch*in_channels*in_height*in_width
        """
        super(PositionalEncoding,self).__init__()
        self.convEvenChannel=nn.Conv2d(in_channels,in_channels+in_channels%2,1)
        self.conv1x1=nn.Conv2d(in_width,1,(1,1),1)
        self.pool=nn.MaxPool2d(2,return_indices=True)

        self.doubleConv=DoubleConv(1,1,3)
        
        self.unpool=nn.MaxUnpool2d(2)
        
        self.convOriginChannel=nn.Conv2d(in_channels+in_channels%2,in_channels,1)
    
    def forward(self,x):
        x_even=self.convEvenChannel(x)
        x1=torch.permute(x_even,(0,3,2,1)) # B,W,H,C
        x2=self.conv1x1(x1) # B,1,H,C
        x3,indices=self.pool(x2)
        x4=self.doubleConv(x3)
        x5=self.unpool(x4,indices=indices)
        x6=torch.permute(x5,(0,3,2,1)) # B,C,W,1
        return self.convOriginChannel(x6)