from ops import *
import torch 
import torch.nn as nn
from torch.autograd import Variable
from base_model import BaseModel
import sys 
sys.path.append("../options/")
from base_options import BaseOptions
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

class wave_vae(BaseModel):
    def __init__(self, opt):
        super(wave_vae, self).__init__(opt)
        self.encoder = encoder()
        if self.opt.gpu_ids:
            self.encoder = self.encoder.cuda()
        self.criterionL1 = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))
    def name(self):
        return 'wave_vae'
    def forward(self,x):
        self.set_input(x) 
        self.reconstruct, self.l, self.h = self.encoder(self.raw_input)
    def backward(self, x):
        self.forward(x)
        self.loss_recon = self.criterionL1(self.reconstruct, self.raw_input)
        self.loss_h = self.h.norm(1)
        self.total_loss = self.loss_recon + self.loss_h
        self.total_loss.backward()
    def update(self, x):
        self.optimizer.zero_grad()
        self.backward(x)
        self.optimizer.step()
    def set_input(self, x):
        self.raw_input = x
        if self.gpu_ids:
            self.raw_input = self.raw_input.cuda()
        self.raw_input = Variable(self.raw_input, requires_grad = False)

def test_encoder():
    e = encoder()
    x = torch.ones((1,3, 224,224))
    x = Variable(x)
    o, l, h = e(x)
    print(o.size())
    print(l.size())
def test_wave_vae():
    opt = BaseOptions().parse()
    wave_vae_model = wave_vae(opt)
    for i in range(10):
        x = torch.zeros((1,3, 224, 224))
        wave_vae_model.update(x)
if __name__ == '__main__':
    test_wave_vae()