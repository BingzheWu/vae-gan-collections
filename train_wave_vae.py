from models.wave_vae import wave_vae
from options.base_options import BaseOptions
from dataset.datasets_factory import make_dataset
import os 
import matplotlib.pyplot as plt
import skimage.io
import skimage.transform
import numpy as np
import torch
from torch.autograd import Variable
def train():
    opt = BaseOptions().parse()
    data_iter = make_dataset(opt, opt.dataset_name, opt.dataroot, opt.annFile)
    wave_vae_model = wave_vae(opt)
    wave_vae_model.load_models(9)
    for epoch in range(opt.num_epochs):
        train_loss = 0
        for batch_idx, data in enumerate(data_iter):
            img, _  = data
            wave_vae_model.update(img)
            train_loss += wave_vae_model.total_loss.data[0]
            loss = wave_vae_model.total_loss.data[0]
            if batch_idx % 100 == 0:
                print(' Train Epoch:{} [{}/{} ({:.0f}%)]\tLoss:{:.6f}'.format(
                    epoch,
                    batch_idx*len(img),
                    len(data_iter.dataset), 100.*batch_idx/len(data_iter),
                    loss
                ))
                wave_vae_model.save_imgs(epoch)
        print("===> Epoch: {} Average Loss: {:.4f}".format(
            epoch, train_loss /len(data_iter)
        ))
        if epoch%1 == 0:
            wave_vae_model.save_models(epoch)
            wave_vae_model.save_imgs(epoch)
def test():
    opt = BaseOptions().parse()
    wave_vae_model = wave_vae(opt)
    wave_vae_model.load_models(9)
    data_dir = '/datasets/webface_eye_220x300/0423975/'
    img_list = os.listdir(data_dir)[:10]
    imgs = []
    for img in img_list:
        rgb_face = skimage.io.imread(os.path.join(data_dir, img))
        rgb_face = skimage.transform.resize(rgb_face, (224, 224))
        imgs.append(rgb_face)
    imgs = np.array(imgs)
    imgs = torch.Tensor(imgs)
    imgs = imgs.permute(0,3,1,2)
    wave_vae_model.forward(imgs)
    wave_vae_model.save_imgs(10)
    print(imgs.shape)


if __name__ == '__main__':
    #train()
    test()