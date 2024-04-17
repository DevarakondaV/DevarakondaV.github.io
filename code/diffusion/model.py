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


class ResBlock(torch.nn.Module):

    def __init__(self, in_feature):
        self._conv1 = torch.nn.Sequential(
            Conv2d(in_features),
        )
        self._conv2 = torch.nn.Sequential(
            Conv2d(in_features)
        )
        return

    def forward(self, x):
        out1 = self._conv1(x)
        out = self._conv2(out1)
        return out + x

class Attention(torch.nn.Module):

    def __init__(
            self,
            dimensions,
            num_heads,
            qkv_bias = False,
            qk_scale = None,
            attn_drop = 0,
            proj_drop = 0
    ):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.dimensions = dimensions
        head_dim = self.dimensions // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = torch.nn.Linear(
            self.dimensions,
            self.dimensions * 3,
            bias=qkv_bias
        )
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.proj = torch.nn.Linear(self.dimensions, self.dimensions)
        self.proj_drop = torch.nn.Dropout(proj_drop)

    def forward(self, x):
        x = x.unsqueeze(0)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k , v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1))*self.scale
        attn = attn.softmax(dim = -1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1,2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class SelfAttentionBlock(torch.nn.Module):

    def __init__(
            self,
            dimensions,
            num_heads,
            qkv_bias=False,
            qk_scale=None,
            drop=0,
            attn_drop=0,
            drop_path=0
    )->None:
        super(SelfAttentionBlock, self).__init__()
        self.flatten = torch.nn.Flatten(1)
        # self.patch_projection = torch.nn.Linear(
        #     patch_size * patch_size, dimensions)
        self.position_embedding = SinPositionEmbedding(dimensions, 10)
        # self.position_embedding.type(torch.bfloat16)
        self.attn = Attention(
            dimensions, num_heads, qkv_bias, qk_scale, attn_drop, drop)
        self.drop_path = torch.nn.Identity()
        self.norm1 = torch.nn.LayerNorm(dimensions)
        self.MLP = torch.nn.Sequential(
            torch.nn.Linear(dimensions, 4 * dimensions),
            torch.nn.GELU(),
            torch.nn.Dropout(drop),
            torch.nn.Linear(4 * dimensions, dimensions),
            torch.nn.Dropout(drop)
        )
        self.norm2 = torch.nn.LayerNorm(dimensions)

    def forward(self, x, t, return_attention=False):
        t_enc = self.position_embedding(t)
        x = self.flatten(x)
        # x_p = self.patch_projection(x)
        encoding = x + t_enc
        y, attn = self.attn(self.norm1(encoding))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.MLP(self.norm2(x)))
        return x


class SinPositionEmbedding(torch.nn.Module):

    def __init__(self, D, T) -> None:
        super().__init__()
        self.D = D
        self.T = T
        self.N = 10000

    def forward(self, t):
        i = torch.arange(self.D // 2)
        embedding = torch.zeros(1, self.D, requires_grad=False)
        embedding[:, 0::2] = torch.sin(t / (self.N ** (2 * i / self.D)))
        embedding[:, 1::2] = torch.cos(t / (self.N ** (2 * i / self.D)))
        return embedding

class UNet(torch.nn.Module):
    def __init__(self, attn_dimension) -> None:
        """
        Unet model for semantic segmentation. Input Size: 3 x 32 x 32
        Output: 3 x 32 x 32
        """


        self.print_shapes=False
        super(UNet, self).__init__()
        self.block = SelfAttentionBlock(attn_dimension, 64)

        self.down_block1=torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, 3, padding=1, bias=False),
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

        self.conv19=torch.nn.Conv2d(64, 1, 1)


    def crop(self, tensor, nHeight, nWidth):
        x, y=tensor.shape[-2:]
        center_x, center_y=x // 2, y // 2
        tX, tY=(center_x - nWidth // 2), center_y - nHeight // 2
        return tensor[:, :, tX:tX + nWidth, tY:tY + nHeight]

    def forward(self, x, t):
        
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

        # print("SSSSSSS B: ", out5.shape)
        x = self.block(out5, t)
        # print("XX: ", x.shape)
        # out5 = x.squeeze(1).unsqueeze(-1).unsqueeze(-1)
        out5 = x.reshape(out5.shape)
        # print("SSSSSSS: ", x.shape)

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
