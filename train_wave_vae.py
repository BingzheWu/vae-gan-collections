from models.wave_vae import wave_vae
from options.base_options import BaseOptions
from dataset.datasets_factory import make_dataset
def train():
    opt = BaseOptions().parse()
    data_iter = make_dataset(opt, opt.dataset_name, opt.dataroot, opt.annFile)
    wave_vae_model = wave_vae(opt)
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
                    loss/len(img)
                ))
        print("===> Epoch: {} Average Loss: {:.4f}".format(
            epoch, train_loss /len(data_iter)
        ))
        if epoch%1 == 0:
            wave_vae_model.save_models(epoch)
            wave_vae_model.save_imgs(epoch)
if __name__ == '__main__':
    train()