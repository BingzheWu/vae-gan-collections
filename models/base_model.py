import os
import torch
class BaseModel():
    def __init(self, opt,):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.is_train = opt.is_train
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoint_dir, opt.name)
    def name(self):
        return 'BaseModel'