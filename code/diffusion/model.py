from pathlib import Path
import torch
from torchvision.transforms.functional import center_crop
import torchvision
import torch.nn.functional as F
import numpy as np
import cv2
import os
import shutil
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torch import optim
import torch.nn.functional as F
from PIL import Image


class SinPositionEmbedding(torch.nn.Module):

    def __init__(self, D) -> None:
        super().__init__()
        self.D = D

    def forward(self, pos, i):
        exp1 = (2 * i) / self.D
        exp2 = ((2*i) + 1) / self.D
        p = 
        p = torch.sin(pos / (1000**exp1))
        p = torch.cos(pos / (1000**exp2))
        return p

class UNet(torch.nn.Module):
    def __init__(self) -> None:
        """
        Unet model for semantic segmentation. Input Size: 3 x 32 x 32
        Output: 3 x 32 x 32
        """


        self.print_shapes=False
        super(UNet, self).__init__()

        self.down_block1=torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 64, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True)
        )
        self.mp1=torch.nn.MaxPool2d(2, 2)

        self.down_block2=torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 128, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
        )
        self.mp2=torch.nn.MaxPool2d(2, 2)

        self.down_block3=torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
        )
        self.mp3=torch.nn.MaxPool2d(2, 2)

        self.down_block4=torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
        )
        self.mp4=torch.nn.MaxPool2d(2, 2)

        self.double_conv1=torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
        )

        self.upConv1=torch.nn.ConvTranspose2d(512, 512, 2, 2)
        self.double_conv2=torch.nn.Sequential(
            torch.nn.Conv2d(1024, 512, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 256, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
        )

        self.upConv2=torch.nn.ConvTranspose2d(256, 256, 2, 2)
        self.double_conv3=torch.nn.Sequential(
            torch.nn.Conv2d(512, 256, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 128, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
        )

        self.upConv3=torch.nn.ConvTranspose2d(128, 128, 2, 2)
        self.double_conv4=torch.nn.Sequential(
            torch.nn.Conv2d(256, 198, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(198),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(198, 64, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
        )

        self.upConv4=torch.nn.ConvTranspose2d(64, 64, 2, 2)
        self.double_conv5=torch.nn.Sequential(
            torch.nn.Conv2d(128, 64, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 64, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
        )

        self.conv19=torch.nn.Conv2d(64, 3, 1)


    def crop(self, tensor, nHeight, nWidth):
        x, y=tensor.shape[-2:]
        center_x, center_y=x // 2, y // 2
        tX, tY=(center_x - nWidth // 2), center_y - nHeight // 2
        return tensor[:, :, tX:tX + nWidth, tY:tY + nHeight]

    def forward(self, x):

        out1=self.down_block1(x)
        x=self.mp1(out1)

        if self.print_shapes:
            print("M1: ", x.shape)

        out2=self.down_block2(x)
        x=self.mp2(out2)

        if self.print_shapes:
            print("M2: ", x.shape)

        out3=self.down_block3(x)
        x=self.mp3(out3)

        if self.print_shapes:
            print("M3: ", x.shape)

        # Block4
        out4=self.down_block4(x)
        x=self.mp4(out4)

        if self.print_shapes:
            print("M4: ", x.shape)

        out5=self.double_conv1(x)

        if self.print_shapes:
            print("Last: ", out5.shape)

        x=self.upConv1(out5)
        diffY=out4.size()[2] - x.size()[2]
        diffX=out4.size()[3] - x.size()[3]
        x=F.pad(x,
            [diffX // 2, diffX - diffX // 2,
              diffY // 2, diffY - diffY // 2]
        )
        x=torch.concat([out4, x], axis=1)
        x=self.double_conv2(x)

        if self.print_shapes:
            print("PUC1: ", x.shape)

        x=self.upConv2(x)
        diffY=out3.size()[2] - x.size()[2]
        diffX=out3.size()[3] - x.size()[3]
        x=F.pad(x,
            [diffX // 2, diffX - diffX // 2,
              diffY // 2, diffY - diffY // 2]
        )
        x=torch.concat([out3, x], axis=1)
        x=self.double_conv3(x)

        if self.print_shapes:
            print("PUC2: ", x.shape)

        x=self.upConv3(x)
        diffY=out2.size()[2] - x.size()[2]
        diffX=out2.size()[3] - x.size()[3]
        x=F.pad(x,
            [diffX // 2, diffX - diffX // 2,
              diffY // 2, diffY - diffY // 2]
        )
        x=torch.concat([out2, x], axis=1)
        x=self.double_conv4(x)

        if self.print_shapes:
            print("PU3: ", x.shape)

        x=self.upConv4(x)
        diffY=out1.size()[2] - x.size()[2]
        diffX=out1.size()[3] - x.size()[3]
        x=F.pad(x,
            [diffX // 2, diffX - diffX // 2,
              diffY // 2, diffY - diffY // 2]
        )
        x=torch.concat([out1, x], axis=1)
        x=self.double_conv5(x)

        out=self.conv19(x)
        if self.print_shapes:
            print("out: ", out.shape)
        return out
