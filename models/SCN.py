
from threading import local
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import time
from typing import Tuple, Dict
from layers.blocks import *
from layers.adain import *

from layers.network import *


class SCN(nn.Module):
    def __init__(self,width, height, num_classes, ndf, z_length, norm, upsample, anatomy_out_channels, num_mask_channels):
        super(SCN, self).__init__()

        self.h = height
        self.w = width
        self.ndf = ndf
        self.z_length = z_length
        self.anatomy_out_channels = anatomy_out_channels
        self.norm = norm
        self.upsample = upsample
        self.num_classes = num_classes
        self.num_mask_channels = num_mask_channels


        self.local_a = LocalA(self.h,self.w,self.ndf,self.anatomy_out_channels,self.z_length,self.num_mask_channels)
        self.spatial_config = Spatial_config(self.anatomy_out_channels,self.z_length,self.num_mask_channels)

    def forward(self, x):
        local_pred = self.local_a(x)
        #print(local_pred.shape) # b c h w d
        spatial_pred = self.spatial_config(local_pred)
        out = local_pred * spatial_pred

        # out = F.softmax(out,dim=1)
        #print(out.shape)
        return out,local_pred,spatial_pred

class LocalA(nn.Module):
    def __init__(self, h,w,ndf, anatomy_out_channels, z_length, num_mask_channels):
        super(LocalA, self).__init__()
        self.anatomy_out_channels = anatomy_out_channels
        self.z_length = z_length
        self.num_mask_channels = num_mask_channels
        self.unet = UNet_3d(h,w,ndf,128,'batchnorm','nearest')
        self.num_classes = 3

        self.conv1 = nn.Conv3d(128,self.num_classes,1)

    def forward(self, x):
        x = self.unet(x)
        out = self.conv1(x)

        return out


# finished
class Spatial_config(nn.Module):
    def __init__(self, anatomy_out_channels, z_length, num_mask_channels):
        super(Spatial_config, self).__init__()
        self.anatomy_out_channels = anatomy_out_channels
        self.z_length = z_length
        self.num_mask_channels = num_mask_channels
        self.downsampling_factor = 4

        # note: padding to the origin size
        self.pool3d = nn.AvgPool3d([self.downsampling_factor] * 3)
        self.conv1 = nn.Conv3d(3,64,5,padding=2)
        # self.conv1 = nn.ConvTranspose3d(3,64,5,dilation=3,output_padding=2)
        # self.conv2 = nn.ConvTranspose3d(64,64,5,dilation=3,output_padding=2)
        self.conv2 = nn.Conv3d(64,64,5,padding=2)
        #self.conv3 = nn.ConvTranspose3d(64,64,5,dilation=3,output_padding=2)
        self.conv3 = nn.Conv3d(64,64,5,padding=2)

        # note: for output
        self.conv4 = nn.Conv3d(64,3,kernel_size=5,padding=2)
        #self.conv4 = nn.ConvTranspose3d(64,3,5,dilation=3,output_padding=2)
        self.up3d = nn.Upsample(scale_factor=4,mode='trilinear')

    def forward(self, x):
    
        x = self.pool3d(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.up3d(x)

        return x



class UNet_3d(nn.Module):
    def __init__(self, width, height, ndf, num_output_channels, normalization, upsample):
        super(UNet_3d, self).__init__()
        """
        UNet autoencoder
        """
        self.h = height
        self.w = width
        self.norm = normalization
        self.ndf = ndf
        self.num_output_channels = num_output_channels
        self.upsample = upsample

        self.encoder_block1 = conv_block_unet_3d(1, self.ndf, 3, 1, 1, self.norm)
        self.encoder_block2 = conv_block_unet_3d(self.ndf, self.ndf * 2, 3, 1, 1, self.norm)
        self.encoder_block3 = conv_block_unet_3d(self.ndf * 2, self.ndf * 4, 3, 1, 1, self.norm)
        self.encoder_block4 = conv_block_unet_3d(self.ndf * 4, self.ndf * 8, 3, 1, 1, self.norm)
        self.maxpool = nn.MaxPool3d(2, 2)

        self.bottleneck = ResConv(self.ndf * 8, self.norm)

        self.decoder_upsample1 = Interpolate(self.h // 8, mode=self.upsample)
        self.decoder_upconv1 = upconv3d(self.ndf * 16, self.ndf * 8, self.norm)
        self.decoder_block1 = conv_block_unet_3d(self.ndf * 16, self.ndf * 8, 3, 1, 1, self.norm)
        self.decoder_upsample2 = Interpolate(self.h // 4, mode=self.upsample)
        self.decoder_upconv2 = upconv3d(self.ndf * 8, self.ndf * 4, self.norm)
        self.decoder_block2 = conv_block_unet_3d(self.ndf * 8, self.ndf * 4, 3, 1, 1, self.norm)
        self.decoder_upsample3 = Interpolate(self.h // 2, mode=self.upsample)
        self.decoder_upconv3 = upconv3d(self.ndf * 4, self.ndf * 2, self.norm)
        self.decoder_block3 = conv_block_unet_3d(self.ndf * 4, self.ndf * 2, 3, 1, 1, self.norm)
        self.decoder_upsample4 = Interpolate(self.h, mode=self.upsample)
        self.decoder_upconv4 = upconv3d(self.ndf * 2, self.ndf, self.norm)
        self.decoder_block4 = conv_block_unet_3d(self.ndf * 2, self.ndf, 3, 1, 1, self.norm)
        self.classifier_conv = nn.Conv3d(self.ndf, self.num_output_channels, 3, 1, 1, 1)

    def forward(self, x):
        #encoder
        s1 = self.encoder_block1(x)
        out = self.maxpool(s1)
        s2 = self.encoder_block2(out)
        out = self.maxpool(s2)
        s3 = self.encoder_block3(out)
        out = self.maxpool(s3)
        s4 = self.encoder_block4(out)
        out = self.maxpool(s4)

        #bottleneck
        out = self.bottleneck(out)

        #decoder
        out = self.decoder_upsample1(out)
        out = self.decoder_upconv1(out)
        #print(s4.shape)
        #print(out.shape)
        out = torch.cat((out, s4), 1)
        out = self.decoder_block1(out)
        out = self.decoder_upsample2(out)
        out = self.decoder_upconv2(out)
        out = torch.cat((out, s3), 1)
        out = self.decoder_block2(out)
        out = self.decoder_upsample3(out)
        out = self.decoder_upconv3(out)
        out = torch.cat((out, s2), 1)
        out = self.decoder_block3(out)
        out = self.decoder_upsample4(out)
        out = self.decoder_upconv4(out)
        out = torch.cat((out, s1), 1)
        out = self.decoder_block4(out)
        out = self.classifier_conv(out)

        return out


