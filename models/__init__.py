from .sdnet import *
from .sdnet2 import *
from .sdnet3 import *
from .muti_sdnet import *
from .weight_init import *

import sys

def get_model(args, params):
    name = args.model_name
    if name == 'Muti_SDNet':
        return Muti_SDNet(args, params['width'], params['height'], params['num_classes'], params['ndf'], params['z_length'],
                     params['norm'], params['upsample'], params['anatomy_out_channels'], params['num_mask_channels'])
    elif name == 'sdnet':
        return SDNet(args, params['width'], params['height'], params['num_classes'], params['ndf'], params['z_length'],
                      params['norm'], params['upsample'], params['anatomy_out_channels'], params['num_mask_channels'])
    elif name == 'sdnet2':
        return SDNet2(args, params['width'], params['height'], params['num_classes'], params['ndf'], params['z_length'],
                      params['norm'], params['upsample'], params['anatomy_out_channels'], params['num_mask_channels'])
    elif name == 'sdnet3':
        return SDNet3(args, params['width'], params['height'], params['num_classes'], params['ndf'], params['z_length'],
                      params['norm'], params['upsample'], params['anatomy_out_channels'], params['num_mask_channels'])
    else:
        print("Could not find the requested model ({})".format(name), file=sys.stderr)