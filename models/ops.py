import torch
import torch.nn as nn


def upsmapleLayer(in_c, out_c, upsample_type = 'basic', padding_type = 'zero'):
    if upsample_type == 'basic':
        upconv = [nn.ConvTranspose2d(in_c, out_c, kernel_size = 4, stride = 2, padding= 1), nn.ReLU()]
    elif upsample_type == 'bilinear':
        upconv = [nn.Upsample(scale_factor = 2, mode = 'bilinear'),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(in_c, out_c, kernel_size = 3, stride = 1, padding =0 )]
    else:
        raise NotImplementedError
    return nn.Sequential(*upconv)
def conv_block(in_c, out_c, k_size, strides, padding, name = 'conv_block', 
    alpha = 0.1, bias = False, batch_norm = True):
    out = nn.Sequential()
    out.add_module(name+'_conv', nn.Conv2d(in_c, out_c, k_size, strides, padding, bias = bias))
    if batch_norm:
        out.add_module(name+'_norm', nn.InstanceNorm2d(out_c, affine = True))
    out.add_module(name+'_activation', nn.LeakyReLU(alpha, inplace = True))
    return out

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.xavier_normal(m.weight.data, gain = 0.02)
    elif classname.find('Linear') !=-1:
        torch.nn.init.xavier_normal(m.weight.data, gain = 0.02)
    elif classname.find('InstanceNorm2d')!=-1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)

