import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import time
from layers.blocks import *
from layers.adain import *
from models.discriminator import *
from models.Encoder import *
from models.Segmentor import *

class AdaINDecoder(nn.Module):
    def __init__(self, anatomy_out_channels):
        super().__init__()
        self.anatomy_out_channels = anatomy_out_channels
        self.conv1 = conv3D_relu(self.anatomy_out_channels, 128, 3, 1, 1)
        self.conv2 = conv3D_relu(128, 64, 3, 1, 1)
        self.conv3 = conv3D_relu(64, 32, 3, 1, 1)
        self.conv4 = conv3D_no_activ(32, 1, 3, 1, 1)

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

class AdaINDecoder_2d(nn.Module):
    def __init__(self, anatomy_out_channels):
        super().__init__()
        self.anatomy_out_channels = anatomy_out_channels
        self.conv1 = conv_relu(self.anatomy_out_channels, 128, 3, 1, 1)
        self.conv2 = conv_relu(128, 64, 3, 1, 1)
        self.conv3 = conv_relu(64, 32, 3, 1, 1)
        self.conv4 = conv_no_activ(32, 1, 3, 1, 1)

    def forward(self, a, z):
        print(a.shape)
        print(z.shape)
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
            # out = torch.cat([a[t,:].view(1, -1, a.shape[-2], a.shape[-1]), x[:,t,:,:].view(-1, 1, x.shape[-2], x.shape[-1])], 1)
            # out = torch.cat([a[:,:,t,:,:].view(a.shape[0], a.shape[1], -1, a.shape[-2], a.shape[-1]), x[:,:,t,:,:].view(x.shape[0],  a.shape[1], -1, x.shape[-2], x.shape[-1])], 1)
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
            # out = torch.cat([a[t,:].view(1, -1, a.shape[-2], a.shape[-1]), x[:,t,:,:].view(-1, 1, x.shape[-2], x.shape[-1])], 1)
            # out = torch.cat([a[:,:,t,:,:].view(a.shape[0], a.shape[1], -1, a.shape[-2], a.shape[-1]), x[:,:,t,:,:].view(x.shape[0],  a.shape[1], -1, x.shape[-2], x.shape[-1])], 1)
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
    def __init__(self, args, width, height, num_classes, ndf, z_length, norm, upsample, anatomy_out_channels, num_mask_channels):
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

        self.m_encoder = MEncoder(self.z_length)

        #self.m_encoder2 = MEncoder2(self.z_length)

        # self.a_encoder1 = AEncoder_3D(self.anatomy_out_channels)
        # self.a_encoder2 = AEncoder_3D(self.anatomy_out_channels)

        self.a_encoder1 = AEncoder_3D(self.anatomy_out_channels)
        self.a_encoder2 = AEncoder_3D(self.anatomy_out_channels)

        # mod: get aencoder 2d
        self.a_encoder3 = AEncoder_2D(self.anatomy_out_channels)
        self.a_encoder4 = AEncoder_2D(self.anatomy_out_channels)

        self.segmentor1 = Segmentor(self.anatomy_out_channels, self.num_classes)

        # mod: get a 2d segmentor
        #self.segmentor2 = Segmentor_2d(self.anatomy_out_channels, self.num_classes)


        # self.decoder = Decoder(self.anatomy_out_channels, self.z_length, self.num_mask_channels)

        # mod: get a 2d decoder
        # self.decoder_2d = Decoder_2d(self.anatomy_out_channels, self.z_length, self.num_mask_channels)

        # self.mask_discriminator = Discriminator(self.num_classes)
        # self.D_Image1 = Discriminator(1)
        # self.D_Image2 = Discriminator(1)

        # note : the us version
        # self.D_Image4 = Discriminator(1)

    #def forward(self, x1, x2, script_type):
    def forward(self, x1, x2, x3, x4, script_type):
        
        # note: x1 64* 64 *64 ;x2 64* 64 *64
        # note: x3 t * 80* 80   ;x4 t * 80* 80 
        # note： at the first time force 80 -> 64
        # note: mr_s, ct, mr_t, us
        # Anatomy Encoders
        x1 = x1.float()
        x2 = x2.float()

        x3 = x3.float()
        x4 = x4.float()

        # mod: b c t h w  to b t c h w
        # x3 = torch.moveaxis(x3,2,1)
        # x4 = torch.moveaxis(x4,2,1)

        # check shape
        # print('check shape')
        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        # print(x4.shape)
        # print('----')

        # anatomic encoder
        a_out_all1 = self.a_encoder1(x1, script_type)
        a_out_all2 = self.a_encoder2(x2, script_type)

        # mod : per slice anatomic encoding 
        # note: use aencoder_2d and cat together

        a_out_all3 = self.a_encoder3(x3, script_type)
        a_out_all4 = self.a_encoder4(x4, script_type)

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

        # mod: watch a out 1 shape
        # print('check shape')
        # print(a_out1.shape)
        # print(a_out2.shape)
        # print(a_out3.shape)
        # print(a_out4.shape)
        # print('----')
        # ^ Modality Encoders
        # note: m encoder keeps the same
        z_out1, mu_out1, logvar_out1 = self.m_encoder(a_out1, x1)
        z_out2, mu_out2, logvar_out2 = self.m_encoder(a_out2, x2)

        #z_out3, mu_out3, logvar_out3 = self.m_encoder2(a_out3, x3)
        #z_out4, mu_out4, logvar_out4 = self.m_encoder2(a_out4, x4)

        # print('check shape')
        # print(z_out1.shape)
        # print(z_out2.shape)
        # print(z_out3.shape)
        # print(z_out4.shape)
        # print('----')

        # segment
        seg_pred1 = self.segmentor1(a_out1, script_type)
        seg_pred2 = self.segmentor1(a_out2, script_type)

        # seg_pred1 = a_out1
        # seg_pred2 = a_out2

        # note: get seg with 2d segmentor
        # seg_pred3 = [self.segmentor2(a_out3[:,:,t,:,:], script_type) for t in range(a_out3.shape[2])]
        # seg_pred4 = [self.segmentor2(a_out4[:,:,t,:,:], script_type) for t in range(a_out4.shape[2])]

        # seg_pred3 = torch.stack(seg_pred3,2)
        # seg_pred4 = torch.stack(seg_pred4,2)

        seg_pred3 = a_out3 
        seg_pred4 = a_out4         

        # print('check shape')
        # print(seg_pred1.shape)
        # print(seg_pred2.shape)
        # print(seg_pred3.shape)
        # print(seg_pred4.shape)
        # print('----')

        # decoder
        # fake_x1 = self.decoder(a_out1, z_out1)
        # fake_x2 = self.decoder(a_out2, z_out2)

        # fake_x1 = fake_x1.view(fake_x1.shape[0], -1, fake_x1.shape[-2], fake_x1.shape[-1])
        # fake_x2 = fake_x2.view(fake_x2.shape[0], -1, fake_x2.shape[-2], fake_x2.shape[-1])


        # #####  decoder - reconstruction

        # fake_x1_m2_def = self.decoder(a_out1, z_out2)
        # fake_x2_m1_def = self.decoder(a_out2, z_out1)
        # fake_x1_m2_def = fake_x1_m2_def.view(fake_x1_m2_def.shape[0], -1, fake_x1_m2_def.shape[-2], fake_x1_m2_def.shape[-1])
        # fake_x2_m1_def = fake_x2_m1_def.view(fake_x2_m1_def.shape[0], -1, fake_x2_m1_def.shape[-2], fake_x2_m1_def.shape[-1])

        # mod: get recons with decoder 2d
        # mod: test 3d version
        # mod: no generate fake 3 4
        # fake_x3 = self.decoder(a_out3, z_out3)
        # fake_x4 = self.decoder(a_out4, z_out4)

        # fake_x3 = fake_x3.view(fake_x3.shape[0], -1, fake_x3.shape[-2], fake_x3.shape[-1])
        # fake_x4 = fake_x4.view(fake_x4.shape[0], -1, fake_x4.shape[-2], fake_x4.shape[-1])

        # fake_x3_m4_def = self.decoder(a_out3, z_out4)
        # fake_x4_m3_def = self.decoder(a_out4, z_out3)

        # fake_x3_m4_def = fake_x3_m4_def.view(fake_x3_m4_def.shape[0], -1, fake_x3_m4_def.shape[-2], fake_x3_m4_def.shape[-1])
        # fake_x4_m3_def = fake_x4_m3_def.view(fake_x4_m3_def.shape[0], -1, fake_x4_m3_def.shape[-2], fake_x4_m3_def.shape[-1])

        # print('check shape')
        # print(fake_x1.shape)
        # print(fake_x2.shape)
        # print(fake_x3.shape)
        # print(fake_x4.shape)
        # print('----')

        # GANs
        # mod: no generate fake 3 4
        # adv_x1 = self.D_Image1(x1)
        # adv_x2 = self.D_Image2(x2)

        # adv_fake_x1 = self.D_Image1(fake_x1)
        # adv_fake_x2 = self.D_Image2(fake_x2)


        # adv_fake_x1_m2_def= self.D_Image2(fake_x1_m2_def)
        # adv_fake_x2_m1_def = self.D_Image1(fake_x2_m1_def)

        # note: the gan is 2d, and reuse the same structure maybe only need a US version.
        
        # adv_x3 = self.D_Image1(x3)

        # adv_x4 = self.D_Image4(x4)


        # adv_fake_x3 = self.D_Image1(fake_x3)

        # adv_fake_x4 = self.D_Image4(fake_x4)


        # adv_fake_x3_m4_def= self.D_Image4(fake_x3_m4_def)
        # adv_fake_x4_m3_def = self.D_Image1(fake_x4_m3_def)

        # print(adv_x1.shape)
        # print(adv_x2.shape)
        # print(adv_x3.shape)
        # print(adv_x4.shape)
        # print('----')


        # note :renew the output
        return a_out_all1, a_out_all2,seg_pred1, seg_pred2,a_out_all3, a_out_all4, seg_pred3, seg_pred4, z_out1, z_out2, a_out1 , a_out2

        return fake_x1, fake_x2, a_out_all1, a_out_all2, seg_pred1, seg_pred2, fake_x1_m2_def, fake_x2_m1_def,adv_x1, adv_x2, adv_fake_x1, adv_fake_x2, adv_fake_x1_m2_def, adv_fake_x2_m1_def, a_out_all3, a_out_all4, seg_pred3, seg_pred4

        return fake_x1, fake_x2, a_out_all1, a_out_all2, seg_pred1, seg_pred2, fake_x1_m2_def, fake_x2_m1_def,adv_x1, adv_x2, adv_fake_x1, adv_fake_x2, adv_fake_x1_m2_def, adv_fake_x2_m1_def, fake_x3, fake_x4, a_out_all3, a_out_all4, seg_pred3, seg_pred4, fake_x3_m4_def, fake_x4_m3_def,adv_x3, adv_x4, adv_fake_x3, adv_fake_x4, adv_fake_x3_m4_def, adv_fake_x4_m3_def

        # return fake_x1, fake_x2, a_out_all1, a_out_all2, seg_pred1, seg_pred2, fake_x1_m2_def, fake_x2_m1_def,adv_x1, adv_x2, adv_fake_x1, adv_fake_x2, adv_fake_x1_m2_def, adv_fake_x2_m1_def