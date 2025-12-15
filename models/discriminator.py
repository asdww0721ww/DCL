import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.blocks import *


class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()
        self.in_channels = in_channels

        self.conv1 = conv_bn_lrelu(self.in_channels, 64, 3, 2, 1)
        self.conv2 = conv_bn_lrelu(64, 128, 3, 2, 1)
        self.conv3 = conv_bn_lrelu(128, 256, 3, 2, 1)
        self.conv4 = conv_bn_lrelu(256, 512, 3, 2, 1)

        # note: need change 64*128
        # self.fc = nn.Linear(512 *4*4, 1)
        # self.fc = nn.Linear(12800, 1)
        self.fc = nn.Linear(8192, 1)

    def forward(self, x):
        x = x.unsqueeze(2)
        T = x.shape[0]
        res_out = []
        for t in range(0, T):
            out = self.conv1(x[t,:,:,:,:])
            out = self.conv2(out)
            out = self.conv3(out)
            out = self.conv4(out)
            out = self.fc(out.view(out.shape[0],-1))
            res_out.append(out)


        del out
        return torch.cat(res_out,0)




class Discriminator_2D(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator_2D, self).__init__()
        self.in_channels = in_channels

        self.conv1 = conv_bn_lrelu(self.in_channels, 64, 3, 2, 1)
        self.conv2 = conv_bn_lrelu(64, 128, 3, 2, 1)
        self.conv3 = conv_bn_lrelu(128, 256, 3, 2, 1)
        self.conv4 = conv_bn_lrelu(256, 512, 3, 2, 1)

        # note: need change 64*128
        # self.fc = nn.Linear(512 *4*4, 1)
        # self.fc = nn.Linear(12800, 1)
        self.fc = nn.Linear(32768, 1)

    def forward(self, x):
        x = x.unsqueeze(2)
        T = x.shape[0]
        res_out = []
        for t in range(0, T):
            out = self.conv1(x[t,:,:,:,:])
            out = self.conv2(out)
            out = self.conv3(out)
            out = self.conv4(out)
            out = self.fc(out.view(out.shape[0],-1))
            res_out.append(out)


        del out
        return torch.cat(res_out,0)



class Img_Discriminator(nn.Module):
    def __init__(self, num_output_channels, num_classes):
        super(Img_Discriminator, self).__init__()
        self.num_output_channels = num_output_channels
        self.num_classes = num_classes + 1  # background as extra class

        self.conv1 = conv_bn_relu(self.num_output_channels, 64, 3, 1, 1)
        self.conv2 = conv_bn_relu(64, 64, 3, 1, 1)
        self.pred = nn.Conv2d(64, self.num_classes, 1, 1, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.pred(out)
        out = F.softmax(out, dim=1)

        return out