import torch.nn as nn

def normalize(x, norm_type):
    if norm_type == 'batchnorm':
        return nn.BatchNorm2d(x)
    else:
        return nn.BatchNorm2d(x) #temp

def normalize3d(x, norm_type):
    if norm_type == 'batchnorm':
        return nn.BatchNorm3d(x)
    else:
        return nn.BatchNorm3d(x) #temp

def deconv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=0):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def deconv(in_channels, out_channels, kernel_size=3, stride=1, padding=0):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
    )

def conv_lrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.LeakyReLU(0.2, inplace=True)
    )

def conv3D_bn_prelu(in_channels, out_channels, ksize=3, pad=1, stride=1, bias=False):
    """
    Convolution Block
    Convolution + Normalization + NonLinear
    """
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=ksize, padding=pad, stride=stride, bias=bias),
        nn.BatchNorm3d(out_channels),
        nn.PReLU()
    )


def conv_bn_lrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True)
    )


def conv3D_bn_lrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm3d(out_channels),
        nn.LeakyReLU(0.2, inplace=True)
    )

def conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )



def conv_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(inplace=True)
    )

def conv3D_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(inplace=True)
    )

def conv_no_activ(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

def conv3D_no_activ(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)

def conv_id_unet(in_channels, out_channels, norm='batchnorm'):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, 1, 0),
        normalize3d(out_channels, norm),
        nn.ReLU(inplace=True)
    )

def upconv(in_channels, out_channels, norm='batchnorm'):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        normalize(out_channels, norm)
    )

def upconv3d(in_channels, out_channels, norm='batchnorm'):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, 3, 1, 1),
        normalize3d(out_channels, norm)
    )

def conv_block_unet(in_channels, out_channels, kernel_size, stride=1, padding=0, norm='batchnorm'):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        normalize(out_channels, norm),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
        normalize(out_channels, norm),
        nn.ReLU(inplace=True),
    )

def conv_block_unet_3d(in_channels, out_channels, kernel_size, stride=1, padding=0, norm='batchnorm'):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
        normalize3d(out_channels, norm),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, kernel_size, stride, padding),
        normalize3d(out_channels, norm),
        nn.ReLU(inplace=True),
    )

def conv_block_unet_last(in_channels, out_channels, kernel_size, stride=1, padding=0, norm='batchnorm'):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        normalize(out_channels, norm),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
        normalize(out_channels, norm),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
    )

def conv_preactivation_relu(in_channels, out_channels, kernel_size=1, stride=1, padding=0, norm='batchnorm'):
    return nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        normalize(out_channels, norm)
    )

def conv_preactivation_relu3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, norm='batchnorm'):
    return nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
        normalize3d(out_channels, norm)
    )


class ResConv(nn.Module):
    def __init__(self, ndf, norm):
        super(ResConv, self).__init__()
        """
        Args:
            ndf: constant number from channels
        """
        self.ndf = ndf
        self.norm = norm
        self.conv1 = conv_preactivation_relu3d(self.ndf, self.ndf * 2, 3, 1, 1, self.norm)
        self.conv2 = conv_preactivation_relu3d(self.ndf * 2 , self.ndf * 2, 3, 1, 1, self.norm)
        self.resconv = conv_preactivation_relu3d(self.ndf , self.ndf * 2, 1, 1, 0, self.norm)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        residual = self.resconv(residual)

        return out + residual


class ResConv2d(nn.Module):
    def __init__(self, ndf, norm):
        super(ResConv2d, self).__init__()
        """
        Args:
            ndf: constant number from channels
        """
        self.ndf = ndf
        self.norm = norm
        self.conv1 = conv_preactivation_relu(self.ndf, self.ndf * 2, 3, 1, 1, self.norm)
        self.conv2 = conv_preactivation_relu(self.ndf * 2 , self.ndf * 2, 3, 1, 1, self.norm)
        self.resconv = conv_preactivation_relu(self.ndf , self.ndf * 2, 1, 1, 0, self.norm)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        residual = self.resconv(residual)

        return out + residual

class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        """
        Args:
            size: expected size after interpolation
            mode: interpolation type (e.g. bilinear, nearest)
        """
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        
    def forward(self, x):
        out = self.interp(x, size=self.size, mode=self.mode) #, align_corners=False
        
        return out