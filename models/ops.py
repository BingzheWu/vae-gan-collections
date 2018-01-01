import torch
import torch.nn as nn


def upsmapleLayer(in_c, out_c, upsample_type = 'basic', padding_type = 'zero'):
    if upsample_type == 'basic':
        upconv = [nn.ConvTranspose2d(in_c, out_c, kernel_size = 4, stride = 2, padding= 1)]
    elif upsample_type == 'bilinear':
        upconv = [nn.Upsample(scale_factor = 3, mode = 'bilinear'),
                    nn.ReflectionPad2d(1),
                    nn.Cov2d(in_c, out_c, kernel_size = 3, stride = 1, padding =0 )]
    else:
        raise NotImplementedError
    return upconv