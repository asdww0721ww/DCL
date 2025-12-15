import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import time
from layers.blocks import *


class Segmentor(nn.Module):
    def __init__(self, num_output_channels, num_classes):
        super(Segmentor, self).__init__()
        self.num_output_channels = num_output_channels
        self.num_classes = num_classes

        self.conv1 = conv3D_bn_prelu(self.num_output_channels, 64, 3, 1, 1)
        self.conv2 = conv3D_bn_prelu(64, 64, 3, 1, 1)
        self.pred = nn.Conv3d(64, self.num_classes, 1, 1, 0)

    def forward(self, x, script_type):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.pred(out)
        if "training".__eq__(script_type):
            out = out
        else:
            out = torch.sigmoid(out)


        return out

class Segmentor_2d(nn.Module):
    def __init__(self, num_output_channels, num_classes):
        super(Segmentor_2d, self).__init__()
        self.num_output_channels = num_output_channels
        self.num_classes = num_classes

        self.conv1 = conv_bn_relu(self.num_output_channels, 64, 3, 1, 1)
        self.conv2 = conv_bn_relu(64, 64, 3, 1, 1)
        self.pred = nn.Conv2d(64, self.num_classes, 1, 1, 0)

    def forward(self, x, script_type):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.pred(out)

        # mod: add softmax layer to get the result for dice
        
        if "training".__eq__(script_type):
            out = torch.sigmoid(out)
        else:
            out = torch.sigmoid(out)


        return out


class Segmentor1(nn.Module):
    def __init__(self, num_output_channels, num_classes):
        super(Segmentor1, self).__init__()
        self.num_output_channels = num_output_channels
        self.num_classes = num_classes

        self.conv1 = conv3D_bn_prelu(self.num_output_channels, 64, 3, 1, 1)
        self.conv2 = conv3D_bn_prelu(64, 64, 3, 1, 1)
        self.pred = nn.Conv3d(64, self.num_classes, 1, 1, 0)

    def forward(self, x, script_type):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.pred(out)
        if "training".__eq__(script_type):
            out = out
        else:
            out = torch.sigmoid(out)


        return out


class Segmentor2(nn.Module):
    def __init__(self, num_output_channels, num_classes):
        super(Segmentor2, self).__init__()
        self.num_output_channels = num_output_channels
        self.num_classes = num_classes

        self.conv1 = conv3D_bn_prelu(self.num_output_channels, 64, 3, 1, 1)
        self.conv2 = conv3D_bn_prelu(64, 64, 3, 1, 1)
        self.pred = nn.Conv3d(64, self.num_classes, 1, 1, 0)

    def forward(self, x, script_type):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.pred(out)
        if "training".__eq__(script_type):
            out = out
        else:
            out = torch.sigmoid(out)


        return out



