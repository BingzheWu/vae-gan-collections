from ops import *
import torch 
import torch.nn as nn
from torch.autograd import Variable

class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        self.encoder1 = nn.Sequential()
        self.encoder1.add_module('encoerder1', conv_block(3, 16, 3, 1, 1, name = 'encoder1'))
        for i in range(2):
            self.encoder1.add_module('encoder2', conv_block(16, 16, 3, 1, 1, name = 'encoder_2'))
        self.l = conv_block(16, 3, 3, 2, 1, name = 'l1')
        self.h = conv_block(16, 3, 3, 2, 1, name = 'l2')
        self.upsample = upsmapleLayer(3, 16, upsample_type = 'bilinear')
        self.encoder_l = nn.Sequential()
        self.encoder_h = nn.Sequential()
        for i in range(3):
            if i == 2:
                self.encoder_l.add_module(str(i), conv_block(16, 3, 3, 1, 1))
            else:
                self.encoder_l.add_module(str(i), conv_block(16, 16, 3, 1,1))
        for i in range(3):
            if i == 2:
                self.encoder_h.add_module(str(i), conv_block(16, 3, 3, 1, 1))
            else:
                self.encoder_h.add_module(str(i), conv_block(16, 16, 3, 1,1))
    def forward(self, x):
        x = self.encoder1(x)
        i_l = self.l(x)
        i_h = self.h(x)
        i_l_ = self.upsample(i_l)
        i_h_ = self.upsample(i_h)
        i_l_ = self.encoder_l(i_l_)
        i_h_ = self.encoder_h(i_h_)
        o = i_h_+i_l_
        return o, i_l, i_h

class wave_vae(nn.Module):
    def __init__(self):
        pass

def test_encoder():
    e = encoder()
    x = torch.zeros((1,3, 224,224))
    x = Variable(x)
    o, l, h = e(x)
    print(o.size())
    print(l.size())
if __name__ == '__main__':
    test_encoder()