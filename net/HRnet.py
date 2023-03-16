import wandb
import torch.utils.data as Data #批处理模块
import torch
from torch.autograd.gradcheck import gradcheck
import torchvision.datasets as dset
import copy
from torchvision import transforms
from torch import nn
import math
import numpy as np
from sklearn import linear_model
import scipy.misc
from matplotlib import pyplot as plt
from PIL import Image
import torch.nn.functional as F
import  os
import net.UNet as UNet
import net.VIT as VIT
class basicBlock(nn.Module):
    def __init__(self,C):
        super(basicBlock, self).__init__()
        self.conv3x3=nn.Sequential(
            nn.Conv2d(C,C,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(C),
            # nn.ReLU(True),
            nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(C),
            # nn.ReLU(True),
        )
        self.R=nn.ReLU(True)
    def forward(self,X):
        X_shortcut = X
        X = self.conv3x3(X)
        out=X+X_shortcut
        out=self.R(out)
        return out

class Stage1Block(nn.Module):
    def __init__(self, C):
        super(Stage1Block, self).__init__()
        self.convUnit = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=1, stride=1),
            nn.BatchNorm2d(C),
            nn.ReLU(True),
            nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(C),
            nn.ReLU(True),
            nn.Conv2d(C, C, kernel_size=1, stride=1),
            nn.BatchNorm2d(C),
            nn.ReLU(True),
        )

    def forward(self, X):
        X_shortcut = X
        X = self.convUnit(X)
        out = X + X_shortcut
        return out
class Stage2Tran(nn.Module):#64*64*256
    def __init__(self):
        super(Stage2Tran, self).__init__()
        self.convUnit1 = nn.Sequential(
            nn.Conv2d(256, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),#64*64*32
        )
        self.convUnit2 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),#32*32*64
        )

    def forward(self, X):
        X1=self.convUnit1(X)
        X2=self.convUnit2(X)
        return X1,X2

class Stage2Fusion(nn.Module):#64*64*256
    def __init__(self):
        super(Stage2Fusion, self).__init__()
        self.R=nn.ReLU(True)
        self.convUnit1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            # nn.ReLU(True),#32*32*64
        )
        self.convUnit2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1, stride=1),
            nn.BatchNorm2d(32),
            # nn.ReLU(True),#64*64*32
            nn.Upsample(scale_factor=2,mode='nearest')
        )

    def forward(self, X1,X2):
        out1=self.R(self.convUnit2(X2)+X1)
        out2=self.R(self.convUnit1(X1)+X2)
        return out1,out2
class Stage3Tran(nn.Module):#64*64*256
    def __init__(self):
        super(Stage3Tran, self).__init__()
        self.convUnit= nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),#64*64*32
        )
    def forward(self, X1,X2):
        X3=self.convUnit(X2)
        return X1,X2,X3

class Stage3Fusion(nn.Module):
    def __init__(self):
        super(Stage3Fusion, self).__init__()
        self.R=nn.ReLU(True)
        self.downUnit1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            # nn.ReLU(True),#32*32*64
        )
        self.downUnit2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            # nn.ReLU(True),#32*32*64
        )
        self.downUnit3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            # nn.ReLU(True),#32*32*64
        )
        self.upUnit1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1, stride=1),
            nn.BatchNorm2d(32),
            # nn.ReLU(True),#64*64*32
            nn.Upsample(scale_factor=2,mode='nearest')
        )
        self.upUnit2 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=1, stride=1),
            nn.BatchNorm2d(32),
            # nn.ReLU(True),#64*64*32
            nn.Upsample(scale_factor=4,mode='nearest')
        )
        self.upUnit3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
            # nn.ReLU(True),#64*64*32
            nn.Upsample(scale_factor=2,mode='nearest')
        )

    def forward(self, X1,X2,X3):
        down1=self.downUnit1(X1)
        down2=self.downUnit2(down1)
        out1=self.R(self.upUnit2(X3)+self.upUnit1(X2)+X1)
        out2=self.R(self.upUnit3(X3)+down1+X2)
        out3=self.R(down2+self.downUnit3(X2)+X3)
        return out1,out2,out3

class Stage4Tran(nn.Module):#64*64*256
    def __init__(self):
        super(Stage4Tran, self).__init__()
        self.convUnit= nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),#64*64*32
        )
    def forward(self, X1,X2,X3):
        X4=self.convUnit(X3)
        return X1,X2,X3,X4

class Stage4Fusion(nn.Module):
    def __init__(self):
        super(Stage4Fusion, self).__init__()
        self.R=nn.ReLU(True)
        self.downUnit1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            # nn.ReLU(True),#32*32*64
        )
        self.downUnit2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            # nn.ReLU(True),#32*32*64
        )
        self.downUnit3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            # nn.ReLU(True),#32*32*64
        )
        self.downUnit4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            # nn.ReLU(True),#32*32*64
        )
        self.downUnit5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            # nn.ReLU(True),#32*32*64
        )
        self.downUnit6 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            # nn.ReLU(True),#32*32*64
        )
        self.upUnit1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1, stride=1),
            nn.BatchNorm2d(32),
            # nn.ReLU(True),#64*64*32
            nn.Upsample(scale_factor=2,mode='nearest')
        )
        self.upUnit2 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=1, stride=1),
            nn.BatchNorm2d(32),
            # nn.ReLU(True),#64*64*32
            nn.Upsample(scale_factor=4,mode='nearest')
        )
        self.upUnit3 = nn.Sequential(
            nn.Conv2d(256, 32, kernel_size=1, stride=1),
            nn.BatchNorm2d(32),
            # nn.ReLU(True),#64*64*32
            nn.Upsample(scale_factor=8,mode='nearest')
        )
        self.upUnit4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
            # nn.ReLU(True),#64*64*32
            nn.Upsample(scale_factor=2,mode='nearest')
        )
        self.upUnit5 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
            # nn.ReLU(True),#64*64*32
            nn.Upsample(scale_factor=4,mode='nearest')
        )
        self.upUnit6 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1),
            nn.BatchNorm2d(128),
            # nn.ReLU(True),#64*64*32
            nn.Upsample(scale_factor=2,mode='nearest')
        )

    def forward(self, X1,X2,X3,X4):
        down1=self.downUnit1(X1)
        down2=self.downUnit2(down1)
        down3=self.downUnit3(down2)

        down4=self.downUnit4(X2)
        down5=self.downUnit5(down4)

        out1=self.R(self.upUnit3(X4)+self.upUnit2(X3)+self.upUnit1(X2)+X1)
        out2=self.R(self.upUnit5(X4)+self.upUnit4(X3)+down1+X2)
        out3=self.R(self.upUnit6(X4)+down4+down2+self.downUnit4(X2)+X3)
        out4=self.R(down5+down3+self.downUnit6(X3)+X4)
        return out1,out2,out3,out4
class  downAndEx(nn.Module):
    def __init__(self,inchannel,outchannel):
        super(downAndEx, self).__init__()
        self.Unit=nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=2, padding=1),  # 64*64
            nn.BatchNorm2d(outchannel),
        )
    def forward(self,X):
        X=self.Unit(X)
        return X
class classifier1(nn.Module):#classic  classifier
    def __init__(self,kinds):
        super(classifier1, self).__init__()
        # 64*64*32 32*32*64 16*16*128 8*8*256
        self.R=nn.ReLU(True)
        self.downUnit1=nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
        )
        self.downUnit2=nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
        )
        self.downUnit3=nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
        )
        self.simple=nn.Sequential(#4*4*128
            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

        )
        self.L=nn.Linear(4*4*128,kinds)
    def forward(self,X1,X2,X3,X4):
        down1=self.R(X2+self.downUnit1(X1))
        down2=self.R(X3+self.downUnit2(down1))
        down3=self.R(X4+self.downUnit3(down2))#8*8*256
        # print(down3.shape)
        # print(self.simple(down3).shape)
        out=self.simple(down3)
        # print(out.shape)
        out=F.softmax(self.L(out.view(out.size(0),-1)))
        return out
class classifier2(nn.Module):#DCt and space information
    def __init__(self,kinds):
        super(classifier2, self).__init__()
        self.Unit1=nn.Sequential(#16*16*128
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # nn.Linear(4*4*128,kinds)
        )#8*8*256
        self.Unit2=nn.Sequential(#8*8*256
            nn.Conv2d(256, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

        )#8*8*256
        self.L=nn.Linear(4*4*64,kinds)
    def forward(self,X1,X2): #X1 16*16*128 X2 8*8*256
        X1=self.Unit1(X1)
        X=X1+X2#8*8*256
        out=F.softmax(self.L(self.Unit2(X).view(X.size(0),-1)))
        return out#Batch*kinds
class classifier3(nn.Module):#only High
    def __init__(self,kinds):
        super(classifier3, self).__init__()
        self.Unit1=nn.Sequential(#64*64*32
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            # nn.Linear(4*4*128,kinds)
        )#8*8*256
        self.L=nn.Linear(4 * 4 *16 , kinds)
    def forward(self,X):
        X=self.Unit1(X)
        # print(X.shape)
        out=F.softmax(self.L(X.view(X.size(0),-1)))
        return out#Batch*kinds

class ChannelChange(nn.Module):
    def __init__(self,inchannel,outchannel):
        super(ChannelChange, self).__init__()
        self.inchannel=inchannel
        self.outchannel=outchannel
        self.Unit=nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(True),
        )
    def forward(self,X):
        return self.Unit(X)
class HRnet(nn.Module):
    def __init__(self,inchannel,kinds):
        super(HRnet, self).__init__()
        self.R=nn.ReLU(True)
        self.Stage0=nn.Sequential(
            nn.Conv2d(inchannel, 32, kernel_size=3, stride=2, padding=1),#128*128
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),#64*64
            nn.BatchNorm2d(64),
            nn.ReLU(True),#64*64*64
        )
        self.Stage1=nn.Sequential(#64*64
            nn.Conv2d(64, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
            # nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            # nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
            # nn.ReLU(True),#64*64*256
        )
        self.Stage1Ex=nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
            # nn.ReLU(True),  # 64*64*256
        )
        self.convUnit1 = Stage1Block(256)
        self.convUnit2 = Stage1Block(256)
        self.convUnit3 = Stage1Block(256)
        self.convUnit4 = Stage1Block(256)
        self.Stage2tran=Stage2Tran()
        self.S2_0convUnit1 = basicBlock(32)
        self.S2_0convUnit2 = basicBlock(32)
        self.S2_0convUnit3 = basicBlock(32)
        self.S2_0convUnit4 = basicBlock(32)
        self.S2_1convUnit1 = basicBlock(64)
        self.S2_1convUnit2 = basicBlock(64)
        self.S2_1convUnit3 = basicBlock(64)
        self.S2_1convUnit4 = basicBlock(64)

        self.stage2fusion=Stage2Fusion()
        self.Stage3tran=Stage3Tran()

        self.S3_0convUnit1 = basicBlock(32)
        self.S3_0convUnit2 = basicBlock(32)
        self.S3_0convUnit3 = basicBlock(32)
        self.S3_0convUnit4 = basicBlock(32)
        self.S3_1convUnit1 = basicBlock(64)
        self.S3_1convUnit2 = basicBlock(64)
        self.S3_1convUnit3 = basicBlock(64)
        self.S3_1convUnit4 = basicBlock(64)
        self.S3_2convUnit1 = basicBlock(128)
        self.S3_2convUnit2 = basicBlock(128)
        self.S3_2convUnit3 = basicBlock(128)
        self.S3_2convUnit4 = basicBlock(128)

        self.Stage3fusion=Stage3Fusion()
        self.Stage4tran=Stage4Tran()

        self.S4_0convUnit1=basicBlock(32)
        self.S4_0convUnit2=basicBlock(32)
        self.S4_0convUnit3=basicBlock(32)
        self.S4_0convUnit4=basicBlock(32)
        self.S4_1convUnit1=basicBlock(64)
        self.S4_1convUnit2=basicBlock(64)
        self.S4_1convUnit3=basicBlock(64)
        self.S4_1convUnit4=basicBlock(64)
        self.S4_2convUnit1=basicBlock(128)
        self.S4_2convUnit2=basicBlock(128)
        self.S4_2convUnit3=basicBlock(128)
        self.S4_2convUnit4=basicBlock(128)
        self.S4_3convUnit1=basicBlock(256)
        self.S4_3convUnit2=basicBlock(256)
        self.S4_3convUnit3=basicBlock(256)
        self.S4_3convUnit4=basicBlock(256)

        self.Stage4fusion = Stage4Fusion()
        self.Classfier1 = classifier1(kinds)
        self.Classfier2 = classifier2(kinds)
        self.Classfier3 = classifier3(kinds)

        self.S3partWeight64 = torch.nn.Parameter(torch.FloatTensor(torch.rand(64,1,1)), requires_grad=True)#64*1*1
        self.S3partWeight128 = torch.nn.Parameter(torch.FloatTensor(torch.rand(128, 1, 1)), requires_grad=True)
        # self.S3partWeight256 = torch.nn.Parameter(torch.FloatTensor(torch.rand(256, 1, 1)), requires_grad=True)
        # self.S4partWeight64 = torch.nn.Parameter(torch.FloatTensor(torch.rand(64, 1, 1)), requires_grad=True)  # 64*1*1
        self.S4partWeight128 = torch.nn.Parameter(torch.FloatTensor(torch.rand(128, 1, 1)), requires_grad=True)
        self.S4partWeight256 = torch.nn.Parameter(torch.FloatTensor(torch.rand(256, 1, 1)), requires_grad=True)
        self.downAndex64 = downAndEx(16,64)
        self.downAndex128 = downAndEx(64,128)
        self.DctBasic1 = basicBlock(16)
        self.DctBasic2 = basicBlock(16)
        self.DctBasic3 = basicBlock(16)
        self.DctBasic4 = basicBlock(16)

        self.DctBasic5 = basicBlock(128)
        self.DctBasic6 = basicBlock(128)
        self.DctBasic7 = basicBlock(128)
        self.DctBasic8 = basicBlock(128)
        self.downAndex256 = downAndEx(128, 256)

        self.ClassWeight = torch.nn.Parameter(torch.FloatTensor(torch.rand(4)), requires_grad=True)
        self.Unetdowm=UNet.UNetdown(n_channels=2)
        self.Unetup=UNet.UNetup(n_classes=2)
        self.Vit=VIT.ViT(in_channels=2,img_size=256,depth=6,n_classes=2)
        self.CC256to32=ChannelChange(inchannel=256,outchannel=32)
        self.CC512to64=ChannelChange(inchannel=512,outchannel=64)
        self.CC512to128=ChannelChange(inchannel=512,outchannel=128)

        self.CC32to256=ChannelChange(inchannel=32,outchannel=256)
        self.CC64to512=ChannelChange(inchannel=64,outchannel=512)
        self.CC128to512=ChannelChange(inchannel=128,outchannel=512)
        self.L1 = nn.Linear(8, 32)
        self.L2 = nn.Linear(32, kinds)

    def forward(self, DCT,X,DctBlockX):#DCTx 1*256*256 DctBlock 64*64*64
        u1,u2,u3,u4,u5=self.Unetdowm(DCT)
        u3=self.CC256to32(u3)
        u4=self.CC512to64(u4)
        u5=self.CC512to128(u5)
        X=self.Stage0(X)
        X1=self.Stage1(X)
        X2=self.Stage1Ex(X)
        # print(X1.shape,X2.shape)
        out=self.R(X1+X2)
        out=self.convUnit1(out)
        out = self.convUnit2(out)
        out = self.convUnit3(out)
        out = self.convUnit4(out)

        s2_0,s2_1=self.Stage2tran(out)
        s2_0=self.S2_0convUnit1(s2_0)
        s2_0 = self.S2_0convUnit2(s2_0)
        s2_0 = self.S2_0convUnit3(s2_0)
        s2_0 = self.S2_0convUnit4(s2_0)
        s2_1=self.S2_1convUnit1(s2_1)
        s2_1 = self.S2_1convUnit2(s2_1)
        s2_1 = self.S2_1convUnit3(s2_1)
        s2_1 = self.S2_1convUnit4(s2_1)
        s2_0,s2_1=self.stage2fusion(s2_0,s2_1)
        s3_0,s3_1,s3_2=self.Stage3tran(s2_0,s2_1)

        s3_0 = self.S3_0convUnit1(s3_0)
        s3_0 = self.S3_0convUnit2(s3_0)
        s3_0 = self.S3_0convUnit3(s3_0)
        s3_0 = self.S3_0convUnit4(s3_0)

        s3_1 = self.S3_1convUnit1(s3_1)
        s3_1 = self.S3_1convUnit2(s3_1)
        s3_1 = self.S3_1convUnit3(s3_1)
        s3_1 = self.S3_1convUnit4(s3_1)

        s3_2 = self.S3_2convUnit1(s3_2)
        s3_2 = self.S3_2convUnit2(s3_2)
        s3_2 = self.S3_2convUnit3(s3_2)
        s3_2 = self.S3_2convUnit4(s3_2)

        s3Dct=self.DctBasic1(DctBlockX) #16*64*64
        s3Dct=self.DctBasic2(s3Dct)
        s3Dct=self.DctBasic3(s3Dct)
        s3Dct=self.DctBasic4(s3Dct)
        s3Dct=self.downAndex64(s3Dct)#64*32*32
        s3Dct32=self.S3partWeight64*s3Dct
        s3Dct=self.downAndex128(s3Dct)#128*16*16
        s3Dct64=self.S3partWeight128*s3Dct

        s3_1=s3_1+s3Dct32
        s3_2=s3_2+s3Dct64

        s3_0=s3_0+u3
        s3_1=s3_1+u4
        s3_2=s3_2+u5

        s4_0,s4_1,s4_2=self.Stage3fusion(s3_0,s3_1,s3_2)

        u3=u3+s4_0
        u4=u4+s4_1
        u5=u5+s4_2
        u3=self.CC32to256(u3)
        u4=self.CC64to512(u4)
        u5=self.CC128to512(u5)
        s4_0,s4_1,s4_2,s4_3=self.Stage4tran(s4_0,s4_1,s4_2)

        s4_0=self.S4_0convUnit1(s4_0)
        s4_0=self.S4_0convUnit2(s4_0)
        s4_0=self.S4_0convUnit3(s4_0)
        s4_0=self.S4_0convUnit4(s4_0)

        s4_1=self.S4_1convUnit1(s4_1)
        s4_1=self.S4_1convUnit2(s4_1)
        s4_1=self.S4_1convUnit3(s4_1)
        s4_1=self.S4_1convUnit4(s4_1)

        s4_2=self.S4_2convUnit1(s4_2)
        s4_2=self.S4_2convUnit2(s4_2)
        s4_2=self.S4_2convUnit3(s4_2)
        s4_2=self.S4_2convUnit4(s4_2)

        s4_3=self.S4_3convUnit1(s4_3)
        s4_3=self.S4_3convUnit2(s4_3)
        s4_3=self.S4_3convUnit3(s4_3)
        s4_3=self.S4_3convUnit4(s4_3)

        s4Dct = self.DctBasic5(s3Dct64)
        s4Dct = self.DctBasic6(s4Dct)
        s4Dct = self.DctBasic7(s4Dct)
        s4Dct = self.DctBasic8(s4Dct)#16*16*128
        s4Dct128 = self.S4partWeight128*s4Dct
        s4Dct256 = self.downAndex256(s4Dct)
        s4Dct256 = self.S4partWeight256*s4Dct256

        s4_2=s4_2+s4Dct128
        s4_3=s4_3+s4Dct256


        s4_0,s4_1,s4_2,s4_3=self.Stage4fusion(s4_0,s4_1,s4_2,s4_3) #64*64*32 32*32*64 16*16*128 8*8*256
        out1=self.Classfier1(s4_0,s4_1,s4_2,s4_3)
        out2=self.Classfier2(s4_2,s4_3)
        # print(s4_0.shape)
        out3=self.Classfier3(s4_0)

        Vitin=self.Unetup(u1,u2,u3,u4,u5)
        vitout=F.softmax(self.Vit(Vitin))
        # out=F.softmax(self.ClassWeight[0]*out1+self.ClassWeight[1]*out2+self.ClassWeight[2]*out3+self.ClassWeight[3]*F.softmax(vitout))
        out=torch.cat([out1,out2,out3,vitout],dim=1)
        out=out.view(out.size(0),-1)
        out=F.softmax(self.L2(self.L1(out)))
        return out1,out2,out3,vitout,out

# class VHRnet(nn.Module):
#     def __init__(self,inchannel,kinds):#默认256大小
#         super(VHRnet, self).__init__()
#         self.HRnet=HRnet(inchannel=inchannel,kinds=kinds)
#         self.Unet=UNet.UNet(n_channels=1,n_classes=2)
#         self.Vit=VIT.ViT(in_channels=2,img_size=256,depth=6,n_classes=2)
#     def forward(self):
#         return 0

if __name__=="__main__":
    net=HRnet(inchannel=1,kinds=2)
    print(net)