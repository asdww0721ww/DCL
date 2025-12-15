import copy
import functools
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import time
from layers.blocks import *
from layers import adain
from models.discriminator import *
from models.Encoder import *
from models.Segmentor import *

from .ddpm.diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler
from .ddpm.diffusion_cond import GaussianDiffusionTrainer as GaussianDiffusionTrainerCond, GaussianDiffusionSampler as GaussianDiffusionSamplerCond
from .ddpm.model import UNet
from .ddpm.model_cond import UNet as UNetCond
from .Adain import adaptive_instance_normalization

class AdaINDecoder(nn.Module):
    def __init__(self, anatomy_out_channels):
        super().__init__()
        self.anatomy_out_channels = anatomy_out_channels
        self.conv1 = conv3D_relu(self.anatomy_out_channels, 128, 3, 1, 1)
        self.conv2 = conv3D_relu(128, 64, 3, 1, 1)
        self.conv3 = conv3D_relu(64, 32, 3, 1, 1)
        self.conv4 = conv3D_no_activ(32, 1, 3, 1, 1)

    def forward(self, a, z):
        out = adain.adaptive_instance_normalization(a, z)
        out = self.conv1(out)
        out = adain.adaptive_instance_normalization(out, z)
        out = self.conv2(out)
        out = adain.adaptive_instance_normalization(out, z)
        out = self.conv3(out)
        out = adain.adaptive_instance_normalization(out, z)
        out = F.tanh(self.conv4(out))

        return out

class AdaINDecoder_2d(nn.Module):
    def __init__(self, anatomy_out_channels):
        super().__init__()
        self.anatomy_out_channels = anatomy_out_channels
        self.conv1 = conv_relu(self.anatomy_out_channels, 128, 3, 1, 1)
        self.conv2 = conv_relu(128, 64, 3, 1, 1)
        self.conv3 = conv_relu(64, 32, 3, 1, 1)
        self.conv4 = conv_no_activ(32, 1, 3, 1, 1)

    def forward(self, a, z):
        # print(a.shape)
        # print(z.shape)
        out = adaptive_instance_normalization(a, z)
        out = self.conv1(out)
        out = adaptive_instance_normalization(out, z)
        out = self.conv2(out)
        out = adaptive_instance_normalization(out, z)
        out = self.conv3(out)
        out = adaptive_instance_normalization(out, z)
        out = F.tanh(self.conv4(out))

        return out

class FiLMDecoder(nn.Module):
    def __init__(self, anatomy_out_channels):
        super().__init__()
        self.anatomy_out_channels = anatomy_out_channels
        self.conv1 = conv_relu(self.anatomy_out_channels, 128, 3, 1, 1)
        self.conv2 = conv_relu(128, 64, 3, 1, 1)
        self.conv3 = conv_relu(64, 32, 3, 1, 1)
        self.conv4 = conv_no_activ(32, 1, 3, 1, 1)

    def forward(self, a, z):
        out = adaptive_instance_normalization(a, z)
        out = self.conv1(out)
        out = adaptive_instance_normalization(out, z)
        out = self.conv2(out)
        out = adaptive_instance_normalization(out, z)
        out = self.conv3(out)
        out = adaptive_instance_normalization(out, z)
        out = F.tanh(self.conv4(out))

        return out

class Decoder(nn.Module):
    def __init__(self, anatomy_out_channels, z_length, num_mask_channels):
        super(Decoder, self).__init__()
        self.anatomy_out_channels = anatomy_out_channels
        self.z_length = z_length
        self.num_mask_channels = num_mask_channels
        self.decoder = AdaINDecoder(self.anatomy_out_channels)
        # self.decoder = FiLMDecoder(self.anatomy_out_channels)

    def forward(self, a, z):
        # t = a.shape[0]
        out = self.decoder(a, z)

        return out

class Decoder_2d(nn.Module):
    def __init__(self, anatomy_out_channels, z_length, num_mask_channels):
        super(Decoder_2d, self).__init__()
        self.anatomy_out_channels = anatomy_out_channels
        self.z_length = z_length
        self.num_mask_channels = num_mask_channels
        self.decoder = AdaINDecoder_2d(self.anatomy_out_channels)
        # self.decoder = FiLMDecoder(self.anatomy_out_channels)

    def forward(self, a, z):
        # t = a.shape[0]
        out = self.decoder(a, z)

        return out



class MEncoder(nn.Module):
    def __init__(self, z_length):
        super(MEncoder, self).__init__()
        """
        VAE encoder to extract intensity (modality) information from the image
        z_length: length of the output vector
        """
        self.z_length = z_length

        self.block1 = conv_bn_lrelu(4, 16, 3, 2, 1)
        self.block2 = conv_bn_lrelu(16, 32, 3, 2, 1)
        self.block3 = conv_bn_lrelu(32, 64, 3, 2, 1)
        self.block4 = conv_bn_lrelu(64, 128, 3, 2, 1)
        # note: need modify
        self.fc = nn.Linear(2048, 32)

        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=128,
            num_layers=2,
            batch_first=True,)

        self.activ = nn.LeakyReLU(0.03, inplace=True)
        self.mu = nn.Linear(128, self.z_length)
        self.logvar = nn.Linear(128, self.z_length)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        
        return mu + eps*std

    def encode(self, x):
        return self.mu(x), self.logvar(x)

    def forward(self, a, x):
        x = x.unsqueeze(1)
        T = x.shape[2]
        seq_out = []
        for t in range(0, T):
            out = torch.cat([a[:,:,t,:,:], x[:,:,t,:,:]], 1)
            out = self.block1(out)
            out = self.block2(out)
            out = self.block3(out)
            out = self.block4(out)
            out = self.fc(out.view(-1, out.shape[1] * out.shape[2] * out.shape[3]))
            # out = self.norm(out)
            out = self.activ(out)
            seq_out.append(out)

        out, (h_n, c_n) = self.lstm(torch.cat(seq_out, 0).unsqueeze(0))
        out = torch.mean(out, dim=1)

        mu, logvar = self.encode(out)
        z = self.reparameterize(mu, logvar)

        del out, seq_out
        return z, mu, logvar



class MEncoder2(nn.Module):
    def __init__(self, z_length):
        super(MEncoder2, self).__init__()
        """
        VAE encoder to extract intensity (modality) information from the image
        z_length: length of the output vector
        """
        self.z_length = z_length

        self.block1 = conv_bn_lrelu(4, 16, 3, 2, 1)
        self.block2 = conv_bn_lrelu(16, 32, 3, 2, 1)
        self.block3 = conv_bn_lrelu(32, 64, 3, 2, 1)
        self.block4 = conv_bn_lrelu(64, 128, 3, 2, 1)
        self.fc = nn.Linear(8192, 32)

        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=128,
            num_layers=2,
            batch_first=True,)

        self.activ = nn.LeakyReLU(0.03, inplace=True)
        self.mu = nn.Linear(128, self.z_length)
        self.logvar = nn.Linear(128, self.z_length)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        
        return mu + eps*std

    def encode(self, x):
        return self.mu(x), self.logvar(x)

    def forward(self, a, x):
        x = x.unsqueeze(1)
        T = x.shape[2]
        seq_out = []
        for t in range(0, T):
            out = torch.cat([a[:,:,t,:,:], x[:,:,t,:,:]], 1)
            out = self.block1(out)
            out = self.block2(out)
            out = self.block3(out)
            out = self.block4(out)
            out = self.fc(out.view(-1, out.shape[1] * out.shape[2] * out.shape[3]))
            # out = self.norm(out)
            out = self.activ(out)
            seq_out.append(out)

        out, (h_n, c_n) = self.lstm(torch.cat(seq_out, 0).unsqueeze(0))
        out = torch.mean(out, dim=1)

        mu, logvar = self.encode(out)
        z = self.reparameterize(mu, logvar)

        del out, seq_out
        return z, mu, logvar

class Muti_SDNet(nn.Module):
    def __init__(self, args, width, height, num_classes, ndf, z_length, norm, upsample, anatomy_out_channels, num_mask_channels,dev0='cuda:0', dev1='cuda:1'):
        super(Muti_SDNet, self).__init__()
        """
        Args:
            width: input width
            height: input height
            upsample: upsampling type (nearest | bilateral)
            num_classes: number of semantice segmentation classes
            z_length: number of modality factors
            anatomy_out_channels: number of anatomy factors
            norm: feature normalization method (BatchNorm)
            ndf: number of feature channels
        """
        self.args = args
        self.h = height
        self.w = width
        self.ndf = ndf
        self.z_length = z_length
        self.anatomy_out_channels = anatomy_out_channels
        self.norm = norm
        self.upsample = upsample
        self.num_classes = num_classes
        self.num_mask_channels = num_mask_channels

        self.dev0 = dev0
        self.dev1 = dev1

        self.m_encoder = MEncoder(self.z_length)
        self.m_encoder2 = MEncoder2(self.z_length)

        self.a_encoder1 = AEncoder_3D(self.anatomy_out_channels)
        # self.a_encoder2 = AEncoder_3D(self.anatomy_out_channels)

        # mod: get aencoder 2d
        self.a_encoder3 = AEncoder_2D(self.anatomy_out_channels)
        # self.a_encoder4 = AEncoder_2D(self.anatomy_out_channels)

        self.segmentor1 = Segmentor(self.anatomy_out_channels, self.num_classes)

        # mod: get a 2d segmentor
        self.segmentor2 = Segmentor_2d(self.anatomy_out_channels, self.num_classes)

        # note: the decoder will be changed with diffusion
        self.decoder = Decoder(self.anatomy_out_channels, self.z_length, self.num_mask_channels)

        # mod: get a 2d decoder
        self.decoder_2d = Decoder_2d(self.anatomy_out_channels, self.z_length, self.num_mask_channels)

        # self.mask_discriminator = Discriminator(self.num_classes)
        self.D_Image1 = Discriminator(1)
        self.D_Image2 = Discriminator(1)

        # note : the us version
        self.D_Image3 = Discriminator_2D(1)
        self.D_Image4 = Discriminator_2D(1)
        

        self.net_model1 = UNet(T=1000,ch=128, ch_mult=[1, 2, 4, 8],attn=[2],num_res_blocks=2, dropout=0.1)
        self.net_model2 = UNet(T=1000,ch=128, ch_mult=[1, 2, 4, 8],attn=[2],num_res_blocks=2, dropout=0.1)
        self.net_model3 = UNet(T=1000,ch=128, ch_mult=[1, 2, 4, 8],attn=[2],num_res_blocks=2, dropout=0.1)
        # self.ema_model = copy.deepcopy(self.net_model1)
        self.trainer1 = GaussianDiffusionTrainer(self.net_model1, 1e-4, 0.02, 1000,)
        self.net_sampler1 = GaussianDiffusionSampler(self.net_model1, 1e-4, 0.02, 1000, self.h,'epsilon', 'fixedlarge')
        self.trainer2 = GaussianDiffusionTrainer(self.net_model2, 1e-4, 0.02, 1000,)
        self.net_sampler2 = GaussianDiffusionSampler(self.net_model2, 1e-4, 0.02, 1000, self.h,'epsilon', 'fixedlarge')
        self.trainer3 = GaussianDiffusionTrainer(self.net_model3, 1e-4, 0.02, 1000,)
        self.net_sampler3 = GaussianDiffusionSampler(self.net_model3, 1e-4, 0.02, 1000, self.h,'epsilon', 'fixedlarge')
        
        # self.net_model1 = UNetCond(T=1000, z_length=self.z_length, anatomy_ch=self.anatomy_out_channels, 
        #                            ch=128, ch_mult=[1, 2, 4, 8], num_res_blocks=2, dropout=0.1)
        # self.net_model2 = UNetCond(T=1000, z_length=self.z_length, anatomy_ch=self.anatomy_out_channels, 
        #                            ch=128, ch_mult=[1, 2, 4, 8], num_res_blocks=2, dropout=0.1)
        # self.net_model3 = UNetCond(T=1000, z_length=self.z_length, anatomy_ch=self.anatomy_out_channels, 
        #                            ch=128, ch_mult=[1, 2, 4, 8], num_res_blocks=2, dropout=0.1)
        # # self.ema_model = copy.deepcopy(self.net_model1)

        # self.trainer1 = GaussianDiffusionTrainerCond(self.net_model1, 1e-4, 0.02, 1000,)
        # self.net_sampler1 = GaussianDiffusionSamplerCond(self.net_model1, 1e-4, 0.02, 1000, self.h,'epsilon', 'fixedlarge')
        # self.trainer2 = GaussianDiffusionTrainerCond(self.net_model2, 1e-4, 0.02, 1000,)
        # self.net_sampler2 = GaussianDiffusionSamplerCond(self.net_model2, 1e-4, 0.02, 1000, self.h,'epsilon', 'fixedlarge')
        # self.trainer3 = GaussianDiffusionTrainerCond(self.net_model3, 1e-4, 0.02, 1000,)
        # self.net_sampler3 = GaussianDiffusionSamplerCond(self.net_model3, 1e-4, 0.02, 1000, self.h,'epsilon', 'fixedlarge')
        # # self.ema_sampler = GaussianDiffusionSampler(self.ema_model, 1e-4, 0.02, 1000)

    def ema(source, target, decay):
        source_dict = source.state_dict()
        target_dict = target.state_dict()
        for key in source_dict.keys():
            target_dict[key].data.copy_(
                target_dict[key].data * decay +
                source_dict[key].data * (1 - decay))
        
    def forward(self, x1, x2, x3, x4, t1, t2, t3, t4,thi,thi2, script_type):
        
        '''
        # note: x1 64* 64 *64 ;x2 64* 64 *64 
        # note: x3 t * 80* 80   ;x4 t * 80* 80 
        # note: at the first time force 80 -> 64
        # note: mr_s, ct, mr_t, us

        '''
        # Anatomy Encoders
        x1 = x1.float()
        x2 = x2.float()

        x3 = x3.float()
        x4 = x4.float()

        # mod: b c t h w  to b t c h w
        # x3 = torch.moveaxis(x3,2,1)
        # x4 = torch.moveaxis(x4,2,1)

        # anatomic encoder
        a_out_all1 = self.a_encoder1(x1, 1, script_type)
        # a_out_all2 = self.a_encoder2(x2, 1, script_type)
        a_out_all2 = self.a_encoder1(x2, 1, script_type)

        # mod : per slice anatomic encoding 
        # note: use aencoder_2d and cat together

        a_out_all3 = self.a_encoder3(x3, script_type)
        a_out_all4 = self.a_encoder3(x4, script_type)

        if "training".__eq__(script_type):
            a_out1 = a_out_all1[0]
            a_out2 = a_out_all2[0]

            #a_out1 = a_out_all1
            #a_out2 = a_out_all2
            a_out3 = a_out_all3[0]
            a_out4 = a_out_all4[0]

        else:
            a_out1 = a_out_all1
            a_out2 = a_out_all2
            a_out3 = a_out_all3
            a_out4 = a_out_all4

        a_out3 = torch.moveaxis(a_out3,2,1)
        a_out4 = torch.moveaxis(a_out4,2,1)

        # ^ Modality Encoders
        # note: m encoder keeps the same
        z_out1, mu_out1, logvar_out1 = self.m_encoder(a_out1, x1)
        z_out2, mu_out2, logvar_out2 = self.m_encoder(a_out2, x2)

        z_out3, mu_out3, logvar_out3 = self.m_encoder2(a_out3, x3)
        z_out4, mu_out4, logvar_out4 = self.m_encoder2(a_out4, x4)

        # segment
        seg_pred1 = self.segmentor1(a_out1, script_type)
        seg_pred2 = self.segmentor1(a_out2, script_type)

        seg_pred3 = [self.segmentor2(a_out3[:,:,t,:,:], script_type) for t in range(a_out3.shape[2])]
        seg_pred4 = [self.segmentor2(a_out4[:,:,t,:,:], script_type) for t in range(a_out4.shape[2])]  

        seg_pred3 = torch.stack(seg_pred3,2)
        seg_pred4 = torch.stack(seg_pred4,2)      

        # # decoder
        fake_x1 = self.decoder(a_out1, z_out1)
        fake_x2 = self.decoder(a_out2, z_out2)

        # fake_x1 = adaptive_instance_normalization(a_out1, z_out1)

        # fake_x2 = adaptive_instance_normalization(a_out2, z_out2)

        # fake_x1t2 = adaptive_instance_normalization(a_out1, z_out2)

        # fake_x2t1 = adaptive_instance_normalization(a_out2, z_out1)
        # print(fake_x2t1.shape)

        fake_x1t2 = self.decoder(a_out1, z_out2)
        fake_x2t1 = self.decoder(a_out2, z_out1)

        tfake_x1 = torch.cat([fake_x1,fake_x1,fake_x1],dim=1)
        tfake_x2 = torch.cat([fake_x2,fake_x2,fake_x2],dim=1)

        tfake_x1t2 = torch.cat([fake_x1t2,fake_x1t2,fake_x1t2],dim=1)
        tfake_x2t1 = torch.cat([fake_x2t1,fake_x2t1,fake_x2t1],dim=1)
        
        fake_x1 = fake_x1.view(fake_x1.shape[0], -1, fake_x1.shape[-2], fake_x1.shape[-1])
        fake_x2 = fake_x2.view(fake_x2.shape[0], -1, fake_x2.shape[-2], fake_x2.shape[-1])
        fake_x1t2 = fake_x1t2.view(fake_x1t2.shape[0], -1, fake_x1t2.shape[-2], fake_x1t2.shape[-1])
        fake_x2t1 = fake_x2t1.view(fake_x2t1.shape[0], -1, fake_x2t1.shape[-2], fake_x2t1.shape[-1])

        adv_x1 = self.D_Image1(x1)
        adv_x2 = self.D_Image2(x2)

        adv_fake_x1 = self.D_Image1(fake_x1)
        adv_fake_x2 = self.D_Image2(fake_x2)

        adv_fake_x1_m2 = self.D_Image2(fake_x1t2)
        adv_fake_x2_m1 = self.D_Image1(fake_x2t1)

        fake_x3 = self.decoder(a_out3, z_out3)
        fake_x4 = self.decoder(a_out4, z_out4)

        tfake_x3 = torch.cat([fake_x3,fake_x3,fake_x3],dim=1)
        tfake_x4 = torch.cat([fake_x4,fake_x4,fake_x4],dim=1)

        fake_x3 = fake_x3.view(fake_x3.shape[0], -1, fake_x3.shape[-2], fake_x3.shape[-1])
        fake_x4 = fake_x4.view(fake_x4.shape[0], -1, fake_x4.shape[-2], fake_x4.shape[-1])

        adv_x3 = self.D_Image3(x3)
        adv_x4 = self.D_Image4(x4)

        adv_fake_x3 = self.D_Image3(fake_x3)
        adv_fake_x4 = self.D_Image4(fake_x4)

        # no condition: use adain to concat anatomy and modality factors
        if "training".__eq__(script_type):
                    
            s_fake_x1 = tfake_x1[:,:,thi,:,:]
            s_fake_x2 = tfake_x2[:,:,thi,:,:]
            s_fake_x1t2 = tfake_x1t2[:,:,thi,:,:]
            s_fake_x2t1 = tfake_x2t1[:,:,thi,:,:]
            
            s_fake_x3 = tfake_x3[:,:,thi2,:,:]
            s_fake_x4 = tfake_x4[:,:,thi2,:,:]
        
            # print(tfake_x1.shape) #torch.Size([1, 3, 64, 64, 64])
            # print(s_fake_x1.shape) #torch.Size([1, 3, 64, 64])
            # print(s_fake_x3.shape) #torch.Size([1, 3, 128, 128])

            xt1_mix, h1_mix, loss1_mix = self.trainer1(s_fake_x1)

            xt2_mix, h2_mix, loss2_mix = self.trainer2(s_fake_x2)

            xt3_mix, h3_mix, loss3_mix = self.trainer1(s_fake_x3)

            xt4_mix, h4_mix, loss4_mix = self.trainer3(s_fake_x4)

            xt1t2_mix, h1t2_mix, loss1t2_mix = self.trainer2(s_fake_x1t2)

            xt2t1_mix, h2t1_mix, loss2t1_mix = self.trainer1(s_fake_x2t1)

            return a_out_all1, a_out_all2, seg_pred1, seg_pred2, a_out_all3, a_out_all4, seg_pred3, seg_pred4, a_out1,a_out2, a_out3, a_out4, z_out1, mu_out1, logvar_out1, z_out2, mu_out2, logvar_out2, loss1t2_mix, loss2t1_mix, fake_x1, fake_x2 , loss1_mix, loss2_mix,fake_x1t2 ,fake_x2t1,loss3_mix,loss4_mix,adv_x1,adv_x2,adv_fake_x1,adv_fake_x2,adv_fake_x1_m2,adv_fake_x2_m1,adv_x3,adv_x4,adv_fake_x3,adv_fake_x4,fake_x3,fake_x4
        
        else:

            h1_mix = torch.randn(size=tfake_x1[:,:,0,:,:].size()).cuda()
            h2_mix = torch.randn(size=tfake_x1[:,:,0,:,:].size()).cuda()
            h1t2_mix = torch.randn(size=tfake_x1t2[:,:,0,:,:].size()).cuda()
            h2t1_mix = torch.randn(size=tfake_x1t2[:,:,0,:,:].size()).cuda()

            h3_mix = torch.randn(size=tfake_x3[:,:,0,:,:].size()).cuda()
            h4_mix = torch.randn(size=tfake_x4[:,:,0,:,:].size()).cuda()

            x1_d = self.net_sampler1(h1_mix)
            x1_d = torch.clamp(x1_d * 0.5 + 0.5,0,1)

            x2_d = self.net_sampler2(h2_mix)
            x2_d = torch.clamp(x2_d * 0.5 + 0.5,0,1)

            x1t2_d = self.net_sampler2(h1t2_mix)
            x1t2_d = torch.clamp(x1t2_d * 0.5 + 0.5,0,1)

            x2t1_d = self.net_sampler1(h2t1_mix)
            x2t1_d = torch.clamp(x2t1_d * 0.5 + 0.5,0,1)

            x3_d = self.net_sampler1(h3_mix)
            x3_d = torch.clamp(x3_d * 0.5 + 0.5,0,1)

            x4_d = self.net_sampler3(h4_mix)
            x4_d = torch.clamp(x4_d * 0.5 + 0.5,0,1)
            
            tfake_x1[:,:,thi,:,:] = x1_d
            tfake_x2[:,:,thi,:,:] = x2_d

            tfake_x1t2[:,:,thi,:,:] = x1t2_d
            tfake_x2t1[:,:,thi,:,:] = x2t1_d

            tfake_x3[:,:,thi2,:,:] = x3_d
            tfake_x4[:,:,thi2,:,:] = x4_d

            fake_x1 = (tfake_x1[:,0,:,:,:]+tfake_x1[:,1,:,:,:]+tfake_x1[:,2,:,:,:])/3
            fake_x2 = (tfake_x2[:,0,:,:,:]+tfake_x2[:,1,:,:,:]+tfake_x2[:,2,:,:,:])/3

            fake_x1t2 = (tfake_x1t2[:,0,:,:,:]+tfake_x1t2[:,1,:,:,:]+tfake_x1t2[:,2,:,:,:])/3
            fake_x2t1 = (tfake_x2t1[:,0,:,:,:]+tfake_x2t1[:,1,:,:,:]+tfake_x2t1[:,2,:,:,:])/3

            fake_x3 = (tfake_x3[:,0,:,:,:]+tfake_x3[:,1,:,:,:]+tfake_x3[:,2,:,:,:])/3
            fake_x4 = (tfake_x4[:,0,:,:,:]+tfake_x4[:,1,:,:,:]+tfake_x4[:,2,:,:,:])/3

            a_out_all1_d = self.a_encoder1(fake_x1,1, script_type)
            a_out_all2_d = self.a_encoder1(fake_x2,1, script_type)

            a_out_all1t2_d = self.a_encoder1(fake_x1t2,1, script_type)
            a_out_all2t1_d = self.a_encoder1(fake_x2t1,1, script_type)

            a_out_all3_d = self.a_encoder3(fake_x3, script_type)
            a_out_all4_d = self.a_encoder3(fake_x4, script_type)
      
        
            a_out1_d = a_out_all1_d
            a_out2_d = a_out_all2_d
            a_out1t2_d = a_out_all1t2_d
            a_out2t1_d = a_out_all2t1_d
            a_out3_d = a_out_all3_d
            a_out4_d = a_out_all4_d
            
            a_out3_d = torch.moveaxis(a_out3_d,2,1)
            a_out4_d = torch.moveaxis(a_out4_d,2,1)
      
            z_out1_d, mu_out1_d, logvar_out1_d = self.m_encoder(a_out1_d, x1)
            z_out2_d, mu_out2_d, logvar_out2_d = self.m_encoder(a_out2_d, x2)

            z_out1t2_d, mu_out1t2_d, logvar_out1t2_d = self.m_encoder(a_out1t2_d, x1)
            z_out2t1_d, mu_out2t1_d, logvar_out2t1_d = self.m_encoder(a_out2t1_d, x2)

            z_out3_d, mu_out3_d, logvar_out3_d = self.m_encoder2(a_out3_d, x3)
            z_out4_d, mu_out4_d, logvar_out4_d = self.m_encoder2(a_out4_d, x4)
            
        
            return a_out_all1, a_out_all2, seg_pred1, seg_pred2, a_out_all3, a_out_all4, seg_pred3, seg_pred4, a_out1,a_out2, a_out3, a_out4, z_out1, mu_out1, logvar_out1, z_out2, mu_out2, logvar_out2, x1t2_d, x2t1_d, a_out_all1_d,a_out_all2_d, a_out_all1t2_d, a_out_all2t1_d, z_out1_d, mu_out1_d, logvar_out1_d, z_out2_d, mu_out2_d, logvar_out2_d, z_out1t2_d, mu_out1t2_d, logvar_out1t2_d , z_out2t1_d, mu_out2t1_d, logvar_out2t1_d,z_out3_d, mu_out3_d, logvar_out3_d, z_out4_d, mu_out4_d, logvar_out4_d

        # # conditional: use diffuser to concat anatomy and modality factors
        # if "training".__eq__(script_type):

        #     # print(x1.shape)       
        #     s_x1 = x1[:,:,:,:]
        #     s_x2 = x2[:,:,:,:]
        #     # print(s_x1.shape)
        #     # s_x1 = x1[:,:,thi,:,:]
        #     # s_x2 = x2[:,:,thi,:,:]
            
        #     # x3 and x4 are [B, T, H, W] or similar. 
        #     s_x3 = x3[:,thi2,:,:].unsqueeze(1)
        #     s_x4 = x4[:,thi2,:,:].unsqueeze(1)
        #     # print(s_x3.shape)

        #     s_a_out1 = a_out_all1[0][:,:,thi,:,:]
        #     s_a_out2 = a_out_all2[0][:,:,thi,:,:]

        #     # print(s_a_out1.shape)
        #     s_a_out3 = a_out3[:,:,thi2,:,:]
        #     s_a_out4 = a_out4[:,:,thi2,:,:]

        #     # z_out1 is [B, z_length]
        
        #     xt1_mix, h1_mix, loss1_mix = self.trainer1(s_x1, s_a_out1, z_out1)

        #     xt2_mix, h2_mix, loss2_mix = self.trainer2(s_x2, s_a_out2, z_out2)

        #     xt3_mix, h3_mix, loss3_mix = self.trainer1(s_x3, s_a_out3, z_out3)

        #     xt4_mix, h4_mix, loss4_mix = self.trainer3(s_x4, s_a_out4, z_out4)

        #     xt1t2_mix, h1t2_mix, loss1t2_mix = self.trainer2(s_fake_x1t2, s_a_out1, z_out2)

        #     xt2t1_mix, h2t1_mix, loss2t1_mix = self.trainer1(s_fake_x2t1, s_a_out2, z_out1)

        #     return a_out_all1, a_out_all2, seg_pred1, seg_pred2, a_out_all3, a_out_all4, seg_pred3, seg_pred4, a_out1,a_out2, a_out3, a_out4, z_out1, mu_out1, logvar_out1, z_out2, mu_out2, logvar_out2, loss1t2_mix, loss2t1_mix, fake_x1, fake_x2 , loss1_mix, loss2_mix,fake_x1t2 ,fake_x2t1,loss3_mix,loss4_mix,adv_x1,adv_x2,adv_fake_x1,adv_fake_x2,adv_fake_x1_m2,adv_fake_x2_m1,adv_x3,adv_x4,adv_fake_x3,adv_fake_x4,fake_x3,fake_x4
        
        # else:

        #     h1_mix = torch.randn(size=tfake_x1[:,:,0,:,:].size()).cuda()
        #     h2_mix = torch.randn(size=tfake_x1[:,:,0,:,:].size()).cuda()
        #     h1t2_mix = torch.randn(size=tfake_x1t2[:,:,0,:,:].size()).cuda()
        #     h2t1_mix = torch.randn(size=tfake_x1t2[:,:,0,:,:].size()).cuda()

        #     h3_mix = torch.randn(size=tfake_x3[:,:,0,:,:].size()).cuda()
        #     h4_mix = torch.randn(size=tfake_x4[:,:,0,:,:].size()).cuda()

        #     s_a_out1 = a_out1[:,:,thi,:,:]
        #     s_a_out2 = a_out2[:,:,thi,:,:]
            
        #     # For 3 and 4
        #     s_a_out3 = a_out3[:,:,thi2,:,:]
        #     s_a_out4 = a_out4[:,:,thi2,:,:]

        #     x1_d = self.net_sampler1(h1_mix, s_a_out1, z_out1)
        #     x1_d = torch.clamp(x1_d * 0.5 + 0.5,0,1)

        #     x2_d = self.net_sampler2(h2_mix, s_a_out2, z_out2)
        #     x2_d = torch.clamp(x2_d * 0.5 + 0.5,0,1)

        #     x1t2_d = self.net_sampler2(h1t2_mix, s_a_out1, z_out2)
        #     x1t2_d = torch.clamp(x1t2_d * 0.5 + 0.5,0,1)

        #     x2t1_d = self.net_sampler1(h2t1_mix, s_a_out2, z_out1)
        #     x2t1_d = torch.clamp(x2t1_d * 0.5 + 0.5,0,1)

        #     x3_d = self.net_sampler1(h3_mix, s_a_out3, z_out3)
        #     x3_d = torch.clamp(x3_d * 0.5 + 0.5,0,1)

        #     x4_d = self.net_sampler3(h4_mix, s_a_out4, z_out4)
        #     x4_d = torch.clamp(x4_d * 0.5 + 0.5,0,1)
            
        #     tfake_x1[:,:,thi,:,:] = x1_d
        #     tfake_x2[:,:,thi,:,:] = x2_d

        #     tfake_x1t2[:,:,thi,:,:] = x1t2_d
        #     tfake_x2t1[:,:,thi,:,:] = x2t1_d

        #     tfake_x3[:,:,thi2,:,:] = x3_d
        #     tfake_x4[:,:,thi2,:,:] = x4_d

        #     fake_x1 = (tfake_x1[:,0,:,:,:]+tfake_x1[:,1,:,:,:]+tfake_x1[:,2,:,:,:])/3
        #     fake_x2 = (tfake_x2[:,0,:,:,:]+tfake_x2[:,1,:,:,:]+tfake_x2[:,2,:,:,:])/3

        #     fake_x1t2 = (tfake_x1t2[:,0,:,:,:]+tfake_x1t2[:,1,:,:,:]+tfake_x1t2[:,2,:,:,:])/3
        #     fake_x2t1 = (tfake_x2t1[:,0,:,:,:]+tfake_x2t1[:,1,:,:,:]+tfake_x2t1[:,2,:,:,:])/3

        #     fake_x3 = (tfake_x3[:,0,:,:,:]+tfake_x3[:,1,:,:,:]+tfake_x3[:,2,:,:,:])/3
        #     fake_x4 = (tfake_x4[:,0,:,:,:]+tfake_x4[:,1,:,:,:]+tfake_x4[:,2,:,:,:])/3

        #     a_out_all1_d = self.a_encoder1(fake_x1,1, script_type)
        #     a_out_all2_d = self.a_encoder1(fake_x2,1, script_type)

        #     a_out_all1t2_d = self.a_encoder1(fake_x1t2,1, script_type)
        #     a_out_all2t1_d = self.a_encoder1(fake_x2t1,1, script_type)

        #     a_out_all3_d = self.a_encoder3(fake_x3, script_type)
        #     a_out_all4_d = self.a_encoder3(fake_x4, script_type)
      
        
        #     a_out1_d = a_out_all1_d
        #     a_out2_d = a_out_all2_d
        #     a_out1t2_d = a_out_all1t2_d
        #     a_out2t1_d = a_out_all2t1_d
        #     a_out3_d = a_out_all3_d
        #     a_out4_d = a_out_all4_d
            
        #     a_out3_d = torch.moveaxis(a_out3_d,2,1)
        #     a_out4_d = torch.moveaxis(a_out4_d,2,1)
      
        #     z_out1_d, mu_out1_d, logvar_out1_d = self.m_encoder(a_out1_d, x1)
        #     z_out2_d, mu_out2_d, logvar_out2_d = self.m_encoder(a_out2_d, x2)

        #     z_out1t2_d, mu_out1t2_d, logvar_out1t2_d = self.m_encoder(a_out1t2_d, x1)
        #     z_out2t1_d, mu_out2t1_d, logvar_out2t1_d = self.m_encoder(a_out2t1_d, x2)

        #     z_out3_d, mu_out3_d, logvar_out3_d = self.m_encoder2(a_out3_d, x3)
        #     z_out4_d, mu_out4_d, logvar_out4_d = self.m_encoder2(a_out4_d, x4)
            
        
        #     return a_out_all1, a_out_all2, seg_pred1, seg_pred2, a_out_all3, a_out_all4, seg_pred3, seg_pred4, a_out1,a_out2, a_out3, a_out4, z_out1, mu_out1, logvar_out1, z_out2, mu_out2, logvar_out2, x1t2_d, x2t1_d, a_out_all1_d,a_out_all2_d, a_out_all1t2_d, a_out_all2t1_d, z_out1_d, mu_out1_d, logvar_out1_d, z_out2_d, mu_out2_d, logvar_out2_d, z_out1t2_d, mu_out1t2_d, logvar_out1t2_d , z_out2t1_d, mu_out2t1_d, logvar_out2t1_d,z_out3_d, mu_out3_d, logvar_out3_d, z_out4_d, mu_out4_d, logvar_out4_d
