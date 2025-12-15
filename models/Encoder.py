import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.blocks import *
from torch.autograd import Variable
from layers.vision import  ResNet101
from torchvision import models
import numpy as np
import monai
from .swinunetr import SwinUNETR,SwinTransformer
from .te_neck import FPN
from .te_heads import PanopticFPNHead
# from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation


class UNet(nn.Module):
    def __init__(self, width, height, ndf, num_output_channels, normalization, upsample):
        super(UNet, self).__init__()
        """
        UNet autoencoder
        """
        self.h = height
        self.w = width
        self.norm = normalization
        self.ndf = ndf
        self.num_output_channels = num_output_channels
        self.upsample = upsample

        self.encoder_block1 = conv_block_unet(1, self.ndf, 3, 1, 1, self.norm)
        self.encoder_block2 = conv_block_unet(self.ndf, self.ndf * 2, 3, 1, 1, self.norm)
        self.encoder_block3 = conv_block_unet(self.ndf * 2, self.ndf * 4, 3, 1, 1, self.norm)
        self.encoder_block4 = conv_block_unet(self.ndf * 4, self.ndf * 8, 3, 1, 1, self.norm)
        self.maxpool = nn.MaxPool2d(2, 2)

        self.bottleneck = ResConv(self.ndf * 8, self.norm)

        self.decoder_upsample1 = Interpolate((self.h // 8, self.w // 8), mode=self.upsample)
        self.decoder_upconv1 = upconv(self.ndf * 16, self.ndf * 8, self.norm)
        self.decoder_block1 = conv_block_unet(self.ndf * 16, self.ndf * 8, 3, 1, 1, self.norm)
        self.decoder_upsample2 = Interpolate((self.h // 4, self.w // 4), mode=self.upsample)
        self.decoder_upconv2 = upconv(self.ndf * 8, self.ndf * 4, self.norm)
        self.decoder_block2 = conv_block_unet(self.ndf * 8, self.ndf * 4, 3, 1, 1, self.norm)
        self.decoder_upsample3 = Interpolate((self.h // 2, self.w // 2), mode=self.upsample)
        self.decoder_upconv3 = upconv(self.ndf * 4, self.ndf * 2, self.norm)
        self.decoder_block3 = conv_block_unet(self.ndf * 4, self.ndf * 2, 3, 1, 1, self.norm)
        self.decoder_upsample4 = Interpolate((self.h, self.w), mode=self.upsample)
        self.decoder_upconv4 = upconv(self.ndf * 2, self.ndf, self.norm)
        self.decoder_block4 = conv_block_unet(self.ndf * 2, self.ndf, 3, 1, 1, self.norm)
        self.classifier_conv = nn.Conv2d(self.ndf, self.num_output_channels, 3, 1, 1, 1)

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


class UNet1(nn.Module):
    def __init__(self, width, height, ndf, num_output_channels, normalization, upsample):
        super(UNet1, self).__init__()
        """
        UNet autoencoder
        """
        self.h = height
        self.w = width
        self.norm = normalization
        self.ndf = ndf
        self.num_output_channels = num_output_channels
        self.upsample = upsample

        self.encoder_block1 = conv_block_unet(1, self.ndf, 3, 1, 1, self.norm)
        self.encoder_block2 = conv_block_unet(self.ndf, self.ndf * 2, 3, 1, 1, self.norm)
        self.encoder_block3 = conv_block_unet(self.ndf * 2, self.ndf * 4, 3, 1, 1, self.norm)
        self.encoder_block4 = conv_block_unet(self.ndf * 4, self.ndf * 8, 3, 1, 1, self.norm)
        self.maxpool = nn.MaxPool2d(2, 2)

        self.bottleneck = ResConv2d(self.ndf * 8, self.norm)

        self.decoder_upsample1 = Interpolate((self.h // 8, self.w // 8), mode=self.upsample)
        self.decoder_upconv1 = upconv(self.ndf * 16, self.ndf * 8, self.norm)
        self.decoder_block1 = conv_block_unet(self.ndf * 16, self.ndf * 8, 3, 1, 1, self.norm)
        self.decoder_upsample2 = Interpolate((self.h // 4, self.w // 4), mode=self.upsample)
        self.decoder_upconv2 = upconv(self.ndf * 8, self.ndf * 4, self.norm)
        self.decoder_block2 = conv_block_unet(self.ndf * 8, self.ndf * 4, 3, 1, 1, self.norm)
        self.decoder_upsample3 = Interpolate((self.h // 2, self.w // 2), mode=self.upsample)
        self.decoder_upconv3 = upconv(self.ndf * 4, self.ndf * 2, self.norm)
        self.decoder_block3 = conv_block_unet(self.ndf * 4, self.ndf * 2, 3, 1, 1, self.norm)
        self.decoder_upsample4 = Interpolate((self.h, self.w), mode=self.upsample)
        self.decoder_upconv4 = upconv(self.ndf * 2, self.ndf, self.norm)
        self.decoder_block4 = conv_block_unet(self.ndf * 2, self.ndf, 3, 1, 1, self.norm)
        self.classifier_conv = nn.Conv2d(self.ndf, self.num_output_channels, 3, 1, 1, 1)

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



class AEncoder(nn.Module):
    def __init__(self, width, height, ndf, num_output_channels, norm, upsample):
        super(AEncoder, self).__init__()
        """
        UNet encoder for the anatomy factors of the image
        num_output_channels: number of spatial (anatomy) factors to encode
        """
        self.width = width
        self.height = height
        self.ndf = ndf
        self.num_output_channels = num_output_channels
        self.norm = norm
        self.upsample = upsample

        self.unet = UNet(self.width, self.height, self.ndf, self.num_output_channels, self.norm, self.upsample)

    def forward(self, x):
        out = self.unet(x)
        out = F.gumbel_softmax(out, hard=True, dim=1)

        return out

class FeatureExtractor(nn.Module):
    '''
    Returns base network to extract visual features from image
    '''
    def __init__(self, args):
        super(FeatureExtractor,self).__init__()
        skip_dims_in = [2048,1024,512,256,64]

        self.base = ResNet101()
        self.base.load_state_dict(models.resnet101(pretrained=True).state_dict())
        # 单通道输入
        w = self.base.conv1.weight
        # print(w.shape)
        self.base.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), bias=True)
        self.base.conv1.weight = torch.nn.Parameter(w[:, :1, 2:5, 2:5])


        self.hidden_size = args.hidden_size
        self.kernel_size = 3
        self.padding = 0 if self.kernel_size == 1 else 1

        self.sk5 = nn.Conv2d(skip_dims_in[0],int(self.hidden_size),self.kernel_size,padding=self.padding)
        self.sk4 = nn.Conv2d(skip_dims_in[1],int(self.hidden_size),self.kernel_size,padding=self.padding)
        self.sk3 = nn.Conv2d(skip_dims_in[2],int(self.hidden_size/2),self.kernel_size,padding=self.padding)
        self.sk2 = nn.Conv2d(skip_dims_in[3],int(self.hidden_size/4),self.kernel_size,padding=self.padding)

        self.bn5 = nn.BatchNorm2d(int(self.hidden_size))
        self.bn4 = nn.BatchNorm2d(int(self.hidden_size))
        self.bn3 = nn.BatchNorm2d(int(self.hidden_size/2))
        self.bn2 = nn.BatchNorm2d(int(self.hidden_size/4))

    def forward(self,x,semseg=False, raw = False):
        x5,x4,x3,x2,x1 = self.base(x)
        x5_skip = F.relu(self.bn5(self.sk5(x5)))
        x4_skip = F.relu(self.bn4(self.sk4(x4)))
        x3_skip = F.relu(self.bn3(self.sk3(x3)))
        x2_skip = F.relu(self.bn2(self.sk2(x2)))

        if semseg:
            return x5
        elif raw:
            return x5, x4, x3, x2, x1
        else:
            #return total_feats
            del x5, x4, x3, x2, x1, x
            return x5_skip, x4_skip, x3_skip, x2_skip

class ConvLSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, args, input_size, hidden_size, kernel_size, padding):
        super(ConvLSTMCell, self).__init__()
        self.use_gpu = args.gpu
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates = nn.Conv2d(input_size +  hidden_size, 4 * hidden_size, kernel_size, padding=padding)
        gpus = [int(id) for id in args.gpu.split(',') if int(id) >= 0]
        self.device = torch.device(
            'cuda:{}'.format(gpus[0]) if torch.cuda.is_available() and len(gpus) > 0 and gpus[0] >= 0 else 'cpu')

    def forward(self, input_, hidden_state_temporal):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if hidden_state_temporal is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            if self.use_gpu != -1:
                hidden_state_temporal = (
                    Variable(torch.zeros(state_size)).to(self.device),
                    Variable(torch.zeros(state_size)).to(self.device)
                )
            else:
                hidden_state_temporal = (
                    Variable(torch.zeros(state_size)),
                    Variable(torch.zeros(state_size))
                )


        prev_hidden_spatial, prev_cell_spatial = hidden_state_temporal

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_hidden_spatial], 1)
        del prev_hidden_spatial, hidden_state_temporal
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = F.sigmoid(in_gate)
        remember_gate = F.sigmoid(remember_gate)
        out_gate = F.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = F.tanh(cell_gate)
        # compute current cell and hidden state
        cell = (remember_gate * prev_cell_spatial) + (in_gate * cell_gate)
        hidden = out_gate * F.tanh(cell)
        del cell_gate, out_gate, remember_gate, in_gate, gates, stacked_inputs
        state = [hidden, cell]

        return state


class Seq_AEncoder(nn.Module):
    """
    The recurrent Encoder
    """
    def __init__(self, args, h, w, anatomy_out_channels):
        super(Seq_AEncoder, self).__init__()
        self.hidden_size = args.hidden_size
        self.anatomy_out_channels = anatomy_out_channels
        self.kernel_size = 3
        padding = 0 if self.kernel_size == 1 else 1

        self.h = h
        self.w = w
        self.dropout_p = args.dropout_p
        self.skip_mode = args.skip_mode

        self.dropout = nn.Dropout(p=self.dropout_p)

        # convlstms have decreasing dimension as width and height increase
        # skip_dims_out = [self.hidden_size, int(self.hidden_size/2),
        #                  int(self.hidden_size/4),int(self.hidden_size/8),int(self.hidden_size/16)]
        skip_dims_out = [self.hidden_size, int(self.hidden_size / 2),
                         int(self.hidden_size / 4), int(self.hidden_size / 8)]

        self.encoder = FeatureExtractor(args)
        # initialize layers for each deconv stage
        self.clstm_list = nn.ModuleList()
        # 4 is the number of deconv steps that we need to reach image size in the output
        for i in range(len(skip_dims_out)):
            if i == 0:
                clstm_in_dim = self.hidden_size
            else:
                clstm_in_dim = skip_dims_out[i - 1]
                if self.skip_mode == 'concat':
                    clstm_in_dim *= 2

            clstm_i = ConvLSTMCell(args, clstm_in_dim, skip_dims_out[i], self.kernel_size, padding=padding)
            self.clstm_list.append(clstm_i)
            del clstm_i , clstm_in_dim

        self.upsample_match = nn.UpsamplingBilinear2d(size=(self.h, self.w))
        self.conv_1 = nn.Conv2d(skip_dims_out[-1], skip_dims_out[-1] // 2, self.kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(skip_dims_out[-1] // 2, affine=True)

        self.conv_out = nn.Conv2d(skip_dims_out[-1] // 2, self.anatomy_out_channels, self.kernel_size, padding=padding)
        self.bn_out = nn.BatchNorm2d(1, affine=True)
        # calculate the dimensionality of classification vector
        # side class activations are taken from the output of the convlstm
        # therefore we need to compute the sum of the dimensionality of outputs
        # from all convlstm layers
        fc_dim = 0
        for sk in skip_dims_out:
            fc_dim += sk

    def forward(self, x1):

        T = x1.shape[1]
        prev_hidden_temporal = []
        outs = []

        for t in range(0, T):
            feats = self.encoder(x1[:,t,:,:].view(-1, 1, x1.shape[-2], x1.shape[-1]))
            clstm_in = feats[0]
            skip_feats = feats[1:]
            hidden_temporal = []
            for i in range(len(skip_feats) + 1):
                # hidden states will be initialized the first time forward is called
                if len(prev_hidden_temporal) == 0:
                    state = self.clstm_list[i](clstm_in, None)
                else:
                    state = self.clstm_list[i](clstm_in, prev_hidden_temporal[i])

                hidden_temporal.append(state)
                hidden = state[0]

                if self.dropout_p > 0:
                    hidden = self.dropout(hidden)

                # apply skip connection
                if i < len(skip_feats):

                    skip_vec = skip_feats[i]
                    upsample = nn.UpsamplingBilinear2d(size=(skip_vec.size()[-2], skip_vec.size()[-1]))
                    hidden = upsample(hidden)
                    # skip connection
                    if self.skip_mode == 'concat':
                        clstm_in = torch.cat([hidden, skip_vec], 1)
                    elif self.skip_mode == 'sum':
                        clstm_in = hidden + skip_vec
                    elif self.skip_mode == 'mul':
                        clstm_in = hidden * skip_vec
                    elif self.skip_mode == 'none':
                        clstm_in = hidden
                    else:
                        raise Exception('Skip connection mode not supported !')
                else:
                    upsample = nn.UpsamplingBilinear2d(size=(hidden.size()[-2] * 2, hidden.size()[-1] * 2))
                    hidden = upsample(hidden)
                    clstm_in = hidden

                del hidden , upsample, state

            upsample_out = self.upsample_match(clstm_in)
            prev_hidden_temporal = hidden_temporal
            out_1 = F.relu(self.bn1(self.conv_1(upsample_out)), inplace=True)
            out_tmp = self.conv_out(out_1)

            outs.append(out_tmp)

            del upsample_out, out_1, clstm_in, skip_feats, feats, out_tmp, hidden_temporal

        del prev_hidden_temporal
        return torch.cat(outs, 0)





def up_sample3d(x, t, mode="trilinear"):
    """
    3D Up Sampling
    """
    return F.interpolate(x, t.size()[2:], mode=mode, align_corners=False)


# todo: need rewrite
def up_sample2d(x, t, mode="trilinear"):
    """
    2D Up Sampling
    """
    return F.interpolate(x, t.size()[2:], mode=mode, align_corners=False)


class ResStage(nn.Module):
    """
    3D Res stage
    """

    def __init__(self, in_chan, out_chan, stride=1):
        super(ResStage, self).__init__()
        self.conv1 = conv3D_bn_prelu(in_chan, out_chan, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_chan, out_chan, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_chan)
        )
        self.non_linear = nn.PReLU()
        self.down_sample = nn.Sequential(
            nn.Conv3d(in_chan, out_chan, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm3d(out_chan))

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        shortcut = self.down_sample(x)
        out = self.non_linear(out + shortcut)

        return out

class ResStage_2d(nn.Module):
    """
    2D Res stage
    """

    def __init__(self, in_chan, out_chan, stride=1):
        super(ResStage_2d, self).__init__()
        self.conv1 = conv_bn_relu(in_chan, out_chan, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chan)
        )
        self.non_linear = nn.PReLU()
        self.down_sample = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_chan))

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        shortcut = self.down_sample(x)
        out = self.non_linear(out + shortcut)

        return out

def down_stage(in_chan, out_chan):
    return nn.Sequential(
        nn.Conv3d(in_chan, out_chan, kernel_size=1, bias=False),
        nn.BatchNorm3d(out_chan),
        nn.PReLU()
    )

def down_stage_2d(in_chan, out_chan):
    return nn.Sequential(
        nn.Conv2d(in_chan, out_chan, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_chan),
        nn.PReLU()
    )


class MixBlock(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(MixBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_chan, out_chan // 4, 3, padding=1, bias=False)
        self.conv3 = nn.Conv3d(in_chan, out_chan // 4, 5, padding=2, bias=False)
        self.conv5 = nn.Conv3d(in_chan, out_chan // 4, 7, padding=3, bias=False)
        self.conv7 = nn.Conv3d(in_chan, out_chan // 4, 9, padding=4, bias=False)
        self.bn1 = nn.BatchNorm3d(out_chan // 4)
        self.bn3 = nn.BatchNorm3d(out_chan // 4)
        self.bn5 = nn.BatchNorm3d(out_chan // 4)
        self.bn7 = nn.BatchNorm3d(out_chan // 4)
        self.nonlinear = nn.PReLU()

    def forward(self, x):
        k1 = self.bn1(self.conv1(x))
        k3 = self.bn3(self.conv3(x))
        k5 = self.bn5(self.conv5(x))
        k7 = self.bn7(self.conv7(x))
        return self.nonlinear(torch.cat((k1, k3, k5, k7), dim=1))


def _SplitChannels(channels, num_groups):
    split_channels = [channels // num_groups for _ in range(num_groups)]
    split_channels[0] += channels - sum(split_channels)
    return split_channels


class FastMixBlock(nn.Module):
    """
    modified from https://github.com/romulus0914/MixNet-PyTorch/blob/master/mixnet.py
    """

    def __init__(self, in_chan, out_chan):
        super(FastMixBlock, self).__init__()
        kernel_size = [1, 3, 5, 7]
        self.num_groups = len(kernel_size)
        self.split_in_channels = _SplitChannels(in_chan, self.num_groups)
        self.split_out_channels = _SplitChannels(out_chan, self.num_groups)

        self.grouped_conv = nn.ModuleList()
        for i in range(self.num_groups):
            self.grouped_conv.append(
                nn.Sequential(
                    nn.Conv3d(
                        self.split_in_channels[i],
                        self.split_out_channels[i],
                        kernel_size[i],
                        stride=1,
                        padding=(kernel_size[i] - 1) // 2,
                        bias=True
                    ),
                    nn.BatchNorm3d(self.split_out_channels[i]),
                    nn.PReLU()
                )
            )

    def forward(self, x):
        if self.num_groups == 1:
            return self.grouped_conv[0](x)

        x_split = torch.split(x, self.split_in_channels, dim=1)
        x = [conv(t) for conv, t in zip(self.grouped_conv, x_split)]
        x = torch.cat(x, dim=1)

        return x



class FastMixBlock_2d(nn.Module):
    """
    modified from https://github.com/romulus0914/MixNet-PyTorch/blob/master/mixnet.py
    """

    def __init__(self, in_chan, out_chan):
        super(FastMixBlock, self).__init__()
        kernel_size = [1, 3, 5, 7]
        self.num_groups = len(kernel_size)
        self.split_in_channels = _SplitChannels(in_chan, self.num_groups)
        self.split_out_channels = _SplitChannels(out_chan, self.num_groups)

        self.grouped_conv = nn.ModuleList()
        for i in range(self.num_groups):
            self.grouped_conv.append(
                nn.Sequential(
                    nn.Conv2d(
                        self.split_in_channels[i],
                        self.split_out_channels[i],
                        kernel_size[i],
                        stride=1,
                        padding=(kernel_size[i] - 1) // 2,
                        bias=True
                    ),
                    nn.BatchNorm2d(self.split_out_channels[i]),
                    nn.PReLU()
                )
            )

    def forward(self, x):
        if self.num_groups == 1:
            return self.grouped_conv[0](x)

        x_split = torch.split(x, self.split_in_channels, dim=1)
        x = [conv(t) for conv, t in zip(self.grouped_conv, x_split)]
        x = torch.cat(x, dim=1)

        return x


class Attention(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(Attention, self).__init__()
        self.mix1 = FastMixBlock(in_chan, out_chan)
        self.conv1 = nn.Conv3d(out_chan, out_chan, kernel_size=1)
        self.mix2 = FastMixBlock(out_chan, out_chan)
        self.conv2 = nn.Conv3d(out_chan, out_chan, kernel_size=1)
        self.norm1 = nn.BatchNorm3d(out_chan)
        self.norm2 = nn.BatchNorm3d(out_chan)
        self.relu = nn.PReLU()

    def forward(self, x):
        shortcut = x
        mix1 = self.conv1(self.mix1(x))
        mix2 = self.mix2(mix1)
        att_map = F.sigmoid(self.conv2(mix2))
        out = self.norm1(x * att_map) + self.norm2(shortcut)
        return self.relu(out), att_map

class Attention_2d(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(Attention_2d, self).__init__()
        self.mix1 = FastMixBlock_2d(in_chan, out_chan)
        self.conv1 = nn.Conv2d(out_chan, out_chan, kernel_size=1)
        self.mix2 = FastMixBlock_2d(out_chan, out_chan)
        self.conv2 = nn.Conv2d(out_chan, out_chan, kernel_size=1)
        self.norm1 = nn.BatchNorm2d(out_chan)
        self.norm2 = nn.BatchNorm2d(out_chan)
        self.relu = nn.PReLU()

    def forward(self, x):
        shortcut = x
        mix1 = self.conv1(self.mix1(x))
        mix2 = self.mix2(mix1)
        att_map = F.sigmoid(self.conv2(mix2))
        out = self.norm1(x * att_map) + self.norm2(shortcut)
        return self.relu(out), att_map

def out_stage(in_chan, out_chan, out_channel):
    return nn.Sequential(
        nn.Conv3d(in_chan, out_chan, kernel_size=3, padding=1),
        nn.BatchNorm3d(out_chan),
        nn.PReLU(),
        nn.Conv3d(out_chan, out_channel, kernel_size=1)
    )

def out_stage_2d(in_chan, out_chan, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_chan),
        nn.PReLU(),
        nn.Conv2d(out_chan, out_channel, kernel_size=1)
    )

class AEncoder_2D1(nn.Module):
    def __init__(self, out_channel=3):
        super(AEncoder_2D1, self).__init__()
        self.init_block = conv_bn_relu(1, 16)
        self.enc1 = ResStage_2d(16, 16, 1)
        self.enc2 = ResStage_2d(16, 32, 2)
        self.enc3 = ResStage_2d(32, 64, 2)
        self.enc4 = ResStage_2d(64, 128, 2)
        self.enc5 = ResStage_2d(128, 128, 2)

        self.dec4 = ResStage_2d(128+128, 64)
        self.dec3 = ResStage_2d(64+64, 32)
        self.dec2 = ResStage_2d(32+32, 16)
        self.dec1 = ResStage_2d(16+16, 16)

        self.down4 = down_stage_2d(64, 16)
        self.down3 = down_stage_2d(32, 16)
        self.down2 = down_stage_2d(16, 16)
        self.down1 = down_stage_2d(16, 16)

        self.mix1 = Attention_2d(16, 16)
        self.mix2 = Attention_2d(16, 16)
        self.mix3 = Attention_2d(16, 16)
        self.mix4 = Attention_2d(16, 16)
        self.mix_out1 = nn.Conv2d(16, out_channel, kernel_size=1)
        self.mix_out2 = nn.Conv2d(16, out_channel, kernel_size=1)
        self.mix_out3 = nn.Conv2d(16, out_channel, kernel_size=1)
        self.mix_out4 = nn.Conv2d(16, out_channel, kernel_size=1)
        self.down_out1 = nn.Conv2d(16, out_channel, kernel_size=1)
        self.down_out2 = nn.Conv2d(16, out_channel, kernel_size=1)
        self.down_out3 = nn.Conv2d(16, out_channel, kernel_size=1)
        self.down_out4 = nn.Conv2d(16, out_channel, kernel_size=1)
        self.out = out_stage_2d(16*4, 64, out_channel)

    def forward(self, x, script_type):
        x = x.unsqueeze(1)
        x = self.init_block(x.type(torch.FloatTensor).cuda())
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)

        dec4 = self.dec4(
            torch.cat((enc4, up_sample3d(enc5, enc4)), dim=1))
        dec3 = self.dec3(
            torch.cat((enc3, up_sample3d(dec4, enc3)), dim=1))
        dec2 = self.dec2(
            torch.cat((enc2, up_sample3d(dec3, enc2)), dim=1))
        dec1 = self.dec1(
            torch.cat((enc1, up_sample3d(dec2, enc1)), dim=1))

        down1 = up_sample3d(self.down1(dec1), x)
        down4 = up_sample3d(self.down4(dec4), x)
        down3 = up_sample3d(self.down3(dec3), x)
        down2 = up_sample3d(self.down2(dec2), x)

        down_out1 = self.down_out1(down1)
        down_out2 = self.down_out2(down2)
        down_out3 = self.down_out3(down3)
        down_out4 = self.down_out4(down4)

        mix1, att1 = self.mix1(down1)
        mix2, att2 = self.mix2(down2)
        mix3, att3 = self.mix3(down3)
        mix4, att4 = self.mix4(down4)

        mix_out1 = self.mix_out1(mix1)
        mix_out2 = self.mix_out2(mix2)
        mix_out3 = self.mix_out3(mix3)
        mix_out4 = self.mix_out4(mix4)
        out = self.out(torch.cat((mix1, mix2, mix3, mix4), dim=1))

        if "training".__eq__(script_type):
            return out, mix_out1, mix_out2, mix_out3, mix_out4, down_out1, down_out2, down_out3, down_out4
        else:
            return out


class AEncoder_2D(nn.Module):
    def __init__(self, out_channel=3,temporal=10):
        super(AEncoder_2D, self).__init__()
        self.net = LSTM_MMUnet(temporal=temporal)

    def forward(self, x, script_type):
        #print(x.shape)
        #x = torch.unsqueeze(x,2)
        out_noseq , out = self.net(x)

        if "training".__eq__(script_type):
            return out, out_noseq 
        else:
            return out

class AEncoder_2D2(nn.Module):
    def __init__(self, out_channel=3,temporal=10):
        super(AEncoder_2D2, self).__init__()
        from .SCN import SCN
        # from .s_miscs import ensure_tuple_rep
        # windowsize = ensure_tuple_rep(7, 3)
        # patchsize = ensure_tuple_rep(2, 3)
        # self.backbone = SwinTransformer(in_chans=1, embed_dim=24, depths=(2, 2, 2, 2),num_heads=(3, 6, 12, 24),window_size=windowsize,patch_size=patchsize)
        # #self.neck = FPN(in_channels=[24,48,96,192,384],out_channels=3,num_outs=5)
        # #self.seg_head = PanopticFPNHead()

        # from .rsis import RSIS
        self.net = SCN(64,64,3,32,8,'batchnorm','nearest',8,3)

    def forward(self, x, script_type):
        x = torch.unsqueeze(x,1)
        #print(x.shape) # torch.Size([3, 1, 10, 128, 128])
        
        out,local_pred,spatial_pred = self.net(x)
        
        #print(out.shape)

        if "training".__eq__(script_type):
            return out,local_pred,spatial_pred
        else:
            return F.softmax(out)


class AEncoder_2D3(nn.Module):
    def __init__(self, out_channel=3,temporal=10):
        super(AEncoder_2D3, self).__init__()
        from .nnformer import nnFormer


        self.net = nnFormer(crop_size=[32,128,128],
                embedding_dim=192,
                input_channels=1, 
                num_classes=3, 
                conv_op=nn.Conv3d, 
                depths=[2,2,2,2],
                num_heads=[6, 12, 24, 24],
                patch_size=[2,4,4],
                window_size=[4,4,4,4],
                deep_supervision=True)

    def forward(self, x, script_type):
        x = torch.unsqueeze(x,1)
        #print(x.shape) # torch.Size([3, 1, 10, 128, 128])
        
        out = self.net(x)
        
        #print(out.shape)

        if "training".__eq__(script_type):
            return out
        else:
            return F.softmax(out)



class AEncoder_3D(nn.Module):
    def __init__(self, out_channel=3):
        super(AEncoder_3D, self).__init__()
        self.init_block1 = conv3D_bn_prelu(1, 16)
        # self.init_block2 = conv3D_bn_prelu(3, 16)
        self.enc1 = ResStage(16, 16, 1)
        self.enc2 = ResStage(16, 32, 2)
        self.enc3 = ResStage(32, 64, 2)
        self.enc4 = ResStage(64, 128, 2)
        self.enc5 = ResStage(128, 128, 2)

        self.dec4 = ResStage(128+128, 64)
        self.dec3 = ResStage(64+64, 32)
        self.dec2 = ResStage(32+32, 16)
        self.dec1 = ResStage(16+16, 16)

        self.down4 = down_stage(64, 16)
        self.down3 = down_stage(32, 16)
        self.down2 = down_stage(16, 16)
        self.down1 = down_stage(16, 16)

        self.mix1 = Attention(16, 16)
        self.mix2 = Attention(16, 16)
        self.mix3 = Attention(16, 16)
        self.mix4 = Attention(16, 16)
        self.mix_out1 = nn.Conv3d(16, out_channel, kernel_size=1)
        self.mix_out2 = nn.Conv3d(16, out_channel, kernel_size=1)
        self.mix_out3 = nn.Conv3d(16, out_channel, kernel_size=1)
        self.mix_out4 = nn.Conv3d(16, out_channel, kernel_size=1)
        self.down_out1 = nn.Conv3d(16, out_channel, kernel_size=1)
        self.down_out2 = nn.Conv3d(16, out_channel, kernel_size=1)
        self.down_out3 = nn.Conv3d(16, out_channel, kernel_size=1)
        self.down_out4 = nn.Conv3d(16, out_channel, kernel_size=1)
        self.out = out_stage(16*4, 64, out_channel)

    def forward(self, x, inch, script_type):    
        if inch == 1:
            if len(x.shape) <5:
                x = x.unsqueeze(1)
            x = self.init_block1(x.type(torch.FloatTensor).cuda())
        # if inch == 3:
            # x = self.init_block2(x.type(torch.FloatTensor).cuda())
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)

        dec4 = self.dec4(
            torch.cat((enc4, up_sample3d(enc5, enc4)), dim=1))
        dec3 = self.dec3(
            torch.cat((enc3, up_sample3d(dec4, enc3)), dim=1))
        dec2 = self.dec2(
            torch.cat((enc2, up_sample3d(dec3, enc2)), dim=1))
        dec1 = self.dec1(
            torch.cat((enc1, up_sample3d(dec2, enc1)), dim=1))

        down1 = up_sample3d(self.down1(dec1), x)
        down4 = up_sample3d(self.down4(dec4), x)
        down3 = up_sample3d(self.down3(dec3), x)
        down2 = up_sample3d(self.down2(dec2), x)

        down_out1 = self.down_out1(down1)
        down_out2 = self.down_out2(down2)
        down_out3 = self.down_out3(down3)
        down_out4 = self.down_out4(down4)

        mix1, att1 = self.mix1(down1)
        mix2, att2 = self.mix2(down2)
        mix3, att3 = self.mix3(down3)
        mix4, att4 = self.mix4(down4)

        mix_out1 = self.mix_out1(mix1)
        mix_out2 = self.mix_out2(mix2)
        mix_out3 = self.mix_out3(mix3)
        mix_out4 = self.mix_out4(mix4)
        out = self.out(torch.cat((mix1, mix2, mix3, mix4), dim=1))

        if "training".__eq__(script_type):
            return out, mix_out1, mix_out2, mix_out3, mix_out4, down_out1, down_out2, down_out3, down_out4
        else:
            return out


class AEncoder_3D2(nn.Module):
    def __init__(self, out_channel=3):
        super(AEncoder_3D2, self).__init__()
        # self.net = SwinUNETR(img_size=(64, 64, 64),in_channels=1,out_channels=out_channel,feature_size=72,use_checkpoint=False,)

        from .unetr import UNETR
        self.net = UNETR(img_shape=(80, 80, 80), input_dim=1, output_dim=3, embed_dim=768, patch_size=16, num_heads=24, dropout=0.1)

    def forward(self, x, script_type):
        x = torch.unsqueeze(x,1)
        out = self.net(x)
        if script_type == "training":
            out = out
        else:
            out = F.softmax(out)
        return out



class AEncoder_3D3(nn.Module):
    def __init__(self, out_channel=3):
        super(AEncoder_3D3, self).__init__()
        from .SCN import SCN

        self.net = SCN(64,64,3,64,8,'batchnorm','nearest',8,3)

    def forward(self, x, script_type):
        x = torch.unsqueeze(x,1)
        #print(x.shape) # torch.Size([b, 1, 64, 64, 64])
        out,local_pred,spatial_pred = self.net(x)
        
        if "training".__eq__(script_type):
            return out,local_pred,spatial_pred
        else:
            return F.softmax(out)



# note: deprecated , no use
class MMUnet(nn.Module):
    """Multi-Modal-Unet"""
    def __init__(self, input_nc, output_nc=3, ngf=32):
        super(MMUnet, self).__init__()
        print('~' * 50)
        print(' ----- Creating MULTI_UNET  ...')
        print('~' * 50)

        self.in_dim = input_nc
        self.out_dim = ngf
        self.final_out_dim = output_nc

        # ~~~ Encoding Paths ~~~~~~ #
        # Encoder (Modality 1) Flair 1
        self.down_1_0 = ConvBlock2d(self.in_dim, self.out_dim)
        self.pool_1_0 = maxpool()
        self.down_2_0 = ConvBlock2d(self.out_dim * 4, self.out_dim * 2)
        self.pool_2_0 = maxpool()
        self.down_3_0 = ConvBlock2d(self.out_dim * 12, self.out_dim * 4)
        self.pool_3_0 = maxpool()
        self.down_4_0 = ConvBlock2d(self.out_dim * 28, self.out_dim * 8)
        self.pool_4_0 = maxpool()

        # Encoder (Modality 2) T1
        self.down_1_1 = ConvBlock2d(self.in_dim, self.out_dim)
        self.pool_1_1 = maxpool()
        self.down_2_1 = ConvBlock2d(self.out_dim * 4, self.out_dim * 2)
        self.pool_2_1 = maxpool()
        self.down_3_1 = ConvBlock2d(self.out_dim * 12, self.out_dim * 4)
        self.pool_3_1 = maxpool()
        self.down_4_1 = ConvBlock2d(self.out_dim * 28, self.out_dim * 8)
        self.pool_4_1 = maxpool()

        # Encoder (Modality 3) T1c
        self.down_1_2 = ConvBlock2d(self.in_dim, self.out_dim)
        self.pool_1_2 = maxpool()
        self.down_2_2 = ConvBlock2d(self.out_dim * 4, self.out_dim * 2)
        self.pool_2_2 = maxpool()
        self.down_3_2 = ConvBlock2d(self.out_dim * 12, self.out_dim * 4)
        self.pool_3_2 = maxpool()
        self.down_4_2 = ConvBlock2d(self.out_dim * 28, self.out_dim * 8)
        self.pool_4_2 = maxpool()

        # Encoder (Modality 4) T2
        self.down_1_3 = ConvBlock2d(self.in_dim, self.out_dim)
        self.pool_1_3 = maxpool()
        self.down_2_3 = ConvBlock2d(self.out_dim * 4, self.out_dim * 2)
        self.pool_2_3 = maxpool()
        self.down_3_3 = ConvBlock2d(self.out_dim * 12, self.out_dim * 4)
        self.pool_3_3 = maxpool()
        self.down_4_3 = ConvBlock2d(self.out_dim * 28, self.out_dim * 8)
        self.pool_4_3 = maxpool()

        # Bridge between Encoder-Decoder
        self.bridge = ConvBlock2d(self.out_dim * 60, self.out_dim * 16) #

        # ~~~ Decoding Path ~~~~~~ #

        self.upLayer1 = UpBlock2d(self.out_dim * 16, self.out_dim * 8)
        self.upLayer2 = UpBlock2d(self.out_dim * 8, self.out_dim * 4)
        self.upLayer3 = UpBlock2d(self.out_dim * 4, self.out_dim * 2)
        self.upLayer4 = UpBlock2d(self.out_dim * 2, self.out_dim * 1)

    def forward(self, input):
        # ~~~~~~ Encoding path ~~~~~~~  #
        i0 = input[:, 0:1, :, :]   # bz * 1  * height * width
        i1 = input[:, 1:2, :, :]
        i2 = input[:, 2:3, :, :]
        i3 = input[:, 3:4, :, :]

        # -----  First Level --------
        down_1_0 = self.down_1_0(i0)  # bz * outdim * height * width
        down_1_1 = self.down_1_1(i1)
        down_1_2 = self.down_1_2(i2)
        down_1_3 = self.down_1_3(i3)

        # -----  Second Level --------
        # Batch Size * (outdim * 4) * (volume_size/2) * (height/2) * (width/2)
        input_2nd_0 = torch.cat((self.pool_1_0(down_1_0),
                                 self.pool_1_1(down_1_1),
                                 self.pool_1_2(down_1_2),
                                 self.pool_1_3(down_1_3)), dim=1)

        input_2nd_1 = torch.cat((self.pool_1_1(down_1_1),
                                 self.pool_1_2(down_1_2),
                                 self.pool_1_3(down_1_3),
                                 self.pool_1_0(down_1_0)), dim=1)

        input_2nd_2 = torch.cat((self.pool_1_2(down_1_2),
                                 self.pool_1_3(down_1_3),
                                 self.pool_1_0(down_1_0),
                                 self.pool_1_1(down_1_1)), dim=1)

        input_2nd_3 = torch.cat((self.pool_1_3(down_1_3),
                                 self.pool_1_0(down_1_0),
                                 self.pool_1_1(down_1_1),
                                 self.pool_1_2(down_1_2)), dim=1)

        down_2_0 = self.down_2_0(input_2nd_0)
        down_2_1 = self.down_2_1(input_2nd_1)
        down_2_2 = self.down_2_2(input_2nd_2)
        down_2_3 = self.down_2_3(input_2nd_3)

        # -----  Third Level --------
        # Max-pool
        down_2_0m = self.pool_2_0(down_2_0)
        down_2_1m = self.pool_2_0(down_2_1)
        down_2_2m = self.pool_2_0(down_2_2)
        down_2_3m = self.pool_2_0(down_2_3)

        input_3rd_0 = torch.cat((down_2_0m, down_2_1m, down_2_2m, down_2_3m), dim=1)
        input_3rd_0 = torch.cat((input_3rd_0, croppCenter(input_2nd_0, input_3rd_0.shape)), dim=1)

        input_3rd_1 = torch.cat((down_2_1m, down_2_2m, down_2_3m, down_2_0m), dim=1)
        input_3rd_1 = torch.cat((input_3rd_1, croppCenter(input_2nd_1, input_3rd_1.shape)), dim=1)

        input_3rd_2 = torch.cat((down_2_2m, down_2_3m, down_2_0m, down_2_1m), dim=1)
        input_3rd_2 = torch.cat((input_3rd_2, croppCenter(input_2nd_2, input_3rd_2.shape)), dim=1)

        input_3rd_3 = torch.cat((down_2_3m, down_2_0m, down_2_1m, down_2_2m), dim=1)
        input_3rd_3 = torch.cat((input_3rd_3, croppCenter(input_2nd_3, input_3rd_3.shape)), dim=1)

        down_3_0 = self.down_3_0(input_3rd_0)
        down_3_1 = self.down_3_1(input_3rd_1)
        down_3_2 = self.down_3_2(input_3rd_2)
        down_3_3 = self.down_3_3(input_3rd_3)

        # -----  Fourth Level --------
        # Max-pool
        down_3_0m = self.pool_3_0(down_3_0)
        down_3_1m = self.pool_3_0(down_3_1)
        down_3_2m = self.pool_3_0(down_3_2)
        down_3_3m = self.pool_3_0(down_3_3)

        input_4th_0 = torch.cat((down_3_0m, down_3_1m, down_3_2m, down_3_3m), dim=1)
        input_4th_0 = torch.cat((input_4th_0, croppCenter(input_3rd_0, input_4th_0.shape)), dim=1)

        input_4th_1 = torch.cat((down_3_1m, down_3_2m, down_3_3m, down_3_0m), dim=1)
        input_4th_1 = torch.cat((input_4th_1, croppCenter(input_3rd_1, input_4th_1.shape)), dim=1)

        input_4th_2 = torch.cat((down_3_2m, down_3_3m, down_3_0m, down_3_1m), dim=1)
        input_4th_2 = torch.cat((input_4th_2, croppCenter(input_3rd_2, input_4th_2.shape)), dim=1)

        input_4th_3 = torch.cat((down_3_3m, down_3_0m, down_3_1m, down_3_2m), dim=1)
        input_4th_3 = torch.cat((input_4th_3, croppCenter(input_3rd_3, input_4th_3.shape)), dim=1)

        down_4_0 = self.down_4_0(input_4th_0)  # 8C
        down_4_1 = self.down_4_1(input_4th_1)
        down_4_2 = self.down_4_2(input_4th_2)
        down_4_3 = self.down_4_3(input_4th_3)

        # ----- Bridge -----
        # Max-pool
        down_4_0m = self.pool_4_0(down_4_0)
        down_4_1m = self.pool_4_0(down_4_1)
        down_4_2m = self.pool_4_0(down_4_2)
        down_4_3m = self.pool_4_0(down_4_3)

        inputBridge = torch.cat((down_4_0m, down_4_1m, down_4_2m, down_4_3m), dim=1)
        inputBridge = torch.cat((inputBridge, croppCenter(input_4th_0, inputBridge.shape)), dim=1)

        bridge = self.bridge(inputBridge)       # bz * 512 * 15 * 15

        # ############################# #
        # ~~~~~~ Decoding path ~~~~~~~  #
        skip_1 = (down_4_0 + down_4_1 + down_4_2 + down_4_3) / 4.0
        skip_2 = (down_3_0 + down_3_1 + down_3_2 + down_3_3) / 4.0
        skip_3 = (down_2_0 + down_2_1 + down_2_2 + down_2_3) / 4.0
        skip_4 = (down_1_0 + down_1_1 + down_1_2 + down_1_3) / 4.0

        x = self.upLayer1(bridge, skip_1)
        x = self.upLayer2(x, skip_2)
        x = self.upLayer3(x, skip_3)
        x = self.upLayer4(x, skip_4)

        return x


class LSTM0(nn.Module):
    def __init__(self, in_c=5, ngf=32):
        super(LSTM0, self).__init__()
        self.conv_gx_lstm0 = nn.Conv2d(in_c+ngf, ngf, kernel_size=3, padding=1)
        self.conv_ix_lstm0 = nn.Conv2d(in_c+ngf, ngf, kernel_size=3, padding=1)
        self.conv_ox_lstm0 = nn.Conv2d(in_c+ngf, ngf, kernel_size=3, padding=1)

    def forward(self, xt):
        """
        :param xt:      bz * 5(num_class) * 240 * 240
        :return:
            hide_1:    bz * ngf(32) * 240 * 240
            cell_1:    bz * ngf(32) * 240 * 240
        """
        gx = self.conv_gx_lstm0(xt)
        ix = self.conv_ix_lstm0(xt)
        ox = self.conv_ox_lstm0(xt)

        gx = torch.tanh(gx)
        ix = torch.sigmoid(ix)
        ox = torch.sigmoid(ox)

        cell_1 = torch.tanh(gx * ix)
        hide_1 = ox * cell_1
        return cell_1, hide_1


class LSTM(nn.Module):
    def __init__(self, in_c=5, ngf=32):
        super(LSTM, self).__init__()
        self.conv_ix_lstm = nn.Conv2d(in_c + ngf, ngf, kernel_size=3, padding=1, bias=True)
        self.conv_ih_lstm = nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False)

        self.conv_fx_lstm = nn.Conv2d(in_c + ngf, ngf, kernel_size=3, padding=1, bias=True)
        self.conv_fh_lstm = nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False)

        self.conv_ox_lstm = nn.Conv2d(in_c + ngf, ngf, kernel_size=3, padding=1, bias=True)
        self.conv_oh_lstm = nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False)

        self.conv_gx_lstm = nn.Conv2d(in_c + ngf, ngf, kernel_size=3, padding=1, bias=True)
        self.conv_gh_lstm = nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False)

    def forward(self, xt, cell_t_1, hide_t_1):
        """
        :param xt:          bz * (5+32) * 240 * 240
        :param hide_t_1:    bz * ngf(32) * 240 * 240
        :param cell_t_1:    bz * ngf(32) * 240 * 240
        :return:
        """
        gx = self.conv_gx_lstm(xt)         # output: bz * ngf(32) * 240 * 240
        gh = self.conv_gh_lstm(hide_t_1)   # output: bz * ngf(32) * 240 * 240
        g_sum = gx + gh
        gt = torch.tanh(g_sum)

        ox = self.conv_ox_lstm(xt)          # output: bz * ngf(32) * 240 * 240
        oh = self.conv_oh_lstm(hide_t_1)    # output: bz * ngf(32) * 240 * 240
        o_sum = ox + oh
        ot = torch.sigmoid(o_sum)

        ix = self.conv_ix_lstm(xt)              # output: bz * ngf(32) * 240 * 240
        ih = self.conv_ih_lstm(hide_t_1)        # output: bz * ngf(32) * 240 * 240
        i_sum = ix + ih
        it = torch.sigmoid(i_sum)

        fx = self.conv_fx_lstm(xt)              # output: bz * ngf(32) * 240 * 240
        fh = self.conv_fh_lstm(hide_t_1)        # output: bz * ngf(32) * 240 * 240
        f_sum = fx + fh
        ft = torch.sigmoid(f_sum)

        cell_t = ft * cell_t_1 + it * gt        # bz * ngf(32) * 240 * 240
        hide_t = ot * torch.tanh(cell_t)            # bz * ngf(32) * 240 * 240

        return cell_t, hide_t


class LSTM_MMUnet(nn.Module):
    def __init__(self, input_nc=1, output_nc=3, ngf=64, temporal=10):
        super(LSTM_MMUnet, self).__init__()
        self.temporal = temporal
        self.mmunet = UNet1(128, 128, ngf, 64,'batchnorm','nearest')
        self.lstm0 = LSTM0(in_c=output_nc , ngf=ngf)
        self.lstm = LSTM(in_c=output_nc , ngf=ngf)

        self.mmout = nn.Conv2d(ngf, output_nc, kernel_size=3, stride=1, padding=1)
        self.out = nn.Conv2d(ngf, output_nc, kernel_size=3, stride=1, padding=1)

        self.dropout = nn.Dropout2d(0.3)

    def forward(self, x):
        """
        :param x:  5D tensor    bz * temporal * c * 240 * 240
        :return:
        """
        # mod :get channel axis
        x = torch.unsqueeze(x,2)
        output = []
        mm_output = []
        cell = None
        hide = None
        #print(x.shape)
        for t in range(self.temporal):
            im_t = x[:, t, :, :, :]                # bz * 4 * 240 * 240
            #print(im_t.shape)
            mm_last = self.mmunet(im_t)              # bz * 32 * 240 * 240
            #print(mm_last.shape)

            # mod : adding dropout for overlapping
            mm_last = self.dropout(mm_last)

            out_t = self.mmout(mm_last)  # bz * 5 * 240 * 240

            
            #out_t = F.sigmoid(out_t)
            #out_t = mm_last   
            #lstm_in = mm_last  
            
            mm_output.append(out_t)
            
            lstm_in = torch.cat((out_t, mm_last), dim=1) # bz * 37 * 240 * 240

            if t == 0:
                cell, hide = self.lstm0(lstm_in)   # bz * ngf(32) * 240 * 240
            else:
                cell, hide = self.lstm(lstm_in, cell, hide)

            out_t = self.out(hide)
            # out_t = F.softmax(out_t,dim=1)
            # out_t = torch.sigmoid(out_t)
            output.append(out_t)

        return torch.stack(mm_output, dim=1), torch.stack(output, dim=1)




def croppCenter(tensorToCrop,finalShape):
    org_shape = tensorToCrop.shape

    diff = np.zeros(2)
    diff[0] = org_shape[2] - finalShape[2]
    diff[1] = org_shape[3] - finalShape[3]

    croppBorders = np.zeros(2,dtype=int)
    croppBorders[0] = int(diff[0]/2)
    croppBorders[1] = int(diff[1]/2)

    return tensorToCrop[:, :,croppBorders[0]:croppBorders[0] + finalShape[2],croppBorders[1]:croppBorders[1] + finalShape[3]]


class ConvBlock2d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock2d, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ConvTrans2d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvTrans2d, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        return x


class UpBlock2d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpBlock2d, self).__init__()
        self.up_conv = ConvTrans2d(in_ch, out_ch)
        self.conv = ConvBlock2d(2 * out_ch, out_ch)

    def forward(self, x, down_features):
        x = self.up_conv(x)
        x = torch.cat([x, down_features], dim=1)
        x = self.conv(x)
        return x


def maxpool():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool


def conv_block(in_dim, out_dim, act_fn, kernel_size=3, stride=1, padding=1, dilation=1):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model


def conv_block_Asym_Inception(in_dim, out_dim, act_fn, kernel_size=3, stride=1, padding=1, dilation=1):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=[kernel_size, 1], padding=tuple([padding, 0]), dilation=(dilation, 1)),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=[1, kernel_size], padding=tuple([0, padding]), dilation=(1, dilation)),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(),
    )
    return model


# TODO: Change order of block: BN + Activation + Conv
def conv_decod_block(in_dim, out_dim, act_fn):
    # sourcery skip: inline-immediately-returned-variable
    model = nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model