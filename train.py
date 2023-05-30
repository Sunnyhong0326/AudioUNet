from opts import make_train_parser
from tqdm import tqdm
import os
import imageio
from dataset.VCTK import VCTKData

from models.AudioUNet import AudioUNet
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import MSELoss
from metric.metrics import AvgPSNR, LSD
from utils import plot_loss_curve

# seed
def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(99)

def train(hparams):
    train_data = VCTKData(hparams, 'train')
    val_data = VCTKData(hparams, 'val')
    
    train_loader = DataLoader(train_data, batch_size = hparams.batch_size, shuffle=True, pin_memory=True, num_workers = 8)
    val_loader = DataLoader(val_data, batch_size = hparams.batch_size, shuffle=False, pin_memory=True, num_workers = 8)
    
    model = AudioUNet(
        hparams.num_filters,
        hparams.num_filter_sizes,
        hparams.up_scale,
        hparams.num_blocks
        )
    
    optim = Adam(model.parameters(), lr = hparams.lr)
    scheduler = CosineAnnealingLR(optimizer = optim, T_max = hparams.num_epochs, eta_min = hparams.lr*0.01)
    criterion = MSELoss()

    model = model.cuda()
    metrics = {'train_loss': [], 'valid_loss': []}

    with tqdm(hparams.num_epochs) as pbar:
        for i_epoch in range(hparams.num_epochs):
            # train
            #############################################
            total_train_loss = 0
            predict_train_result = []
            gt_hr_train = []
            model.train()
            for data in iter(train_loader):
                batch_train_lr = data['lr'].cuda()
                batch_train_hr = data['hr'].cuda()
                # forward propagation
                predict_train_hr = model(batch_train_lr)
                # clear grad buffer
                model.zero_grad()
                # compute loss
                loss = criterion(predict_train_hr, batch_train_hr)
                # accumulate loss
                total_train_loss += loss.detach().item()
                # backward to compute gradient
                loss.backward()
                # update model's weight
                optim.step()
                predict_train_result.append(predict_train_hr.detach().cpu().numpy())
                gt_hr_train.append(batch_train_hr.detach().cpu().numpy())

            avg_train_loss = total_train_loss / len(train_data)
            metrics['train_loss'].append(avg_train_loss)
            #############################################

            #################################################
            # validation
            model.eval()
            total_val_loss = 0
            predict_val_result = []
            gt_hr_val = []

            with torch.no_grad():
                for data in iter(val_loader):
                    batch_val_lr = data['lr'].cuda()
                    batch_val_hr = data['hr'].cuda()
                    predict_val_hr = model(batch_val_lr)
                    # compute loss
                    loss = criterion(predict_val_hr, batch_val_hr)
                    total_val_loss += loss.detach().item()
                    # Save predicted high resolution audio and actual high resolution audio 
                    # and convert to numpy from GPU to CPU
                    predict_val_result.append(predict_val_hr.detach().cpu().numpy())
                    gt_hr_val.append(batch_val_hr.detach().cpu().numpy())

            avg_val_loss = total_val_loss / len(val_data)
            #####################################################################
            scheduler.step()

            # update my process
            pbar.set_description(f'Epoch [{i_epoch+1}/{hparams.num_epochs}]') # prefix str
            
            # use pbar.set_postfix() to setting infomation : train_loss , val_loss , val_psnr and val_LSD
            pbar.set_postfix(
                Avg_Train_Loss = avg_train_loss,
                Avg_Train_PSNR = AvgPSNR(predict_train_result, gt_hr_train),
                Avg_Val_Loss = avg_val_loss,
                Avg_Val_PSNR = AvgPSNR(predict_val_result, gt_hr_val),
                # TODO LSD loss
                )
            pbar.update(1)
            
            # TODO: plot spectrogram
            # Every 50 epochs save the spectrum image
            # if(i_epoch % 50 == 0):
            #     val_spectrum = None
            #     val_img = (all_predict_rgb.numpy()*255).astype(np.uint8).reshape(150, 150, 3)
            #     imageio.imwrite(f'results/{hparams.exp}/val_result_epoch:val_{i_epoch}.png', val_img)

            # saving the checkpoint   
            torch.save({
                'model_state_dict': model.state_dict(),
                'patch_size': hparams.patch_size,
                'epoch': hparams.num_epochs,
                'num_blocks': hparams.num_blocks,
                'num_filters': hparams.num_filters,
                'num_filter_sizes': hparams.num_filter_sizes,
                'up_scale': hparams.up_scale
                }
                , f'ckpts/{hparams.exp}.pth')
    plot_loss_curve(os.path.join(hparams.result_path, hparams.exp), metrics, hparams.num_epochs)

    
if __name__=='__main__':
    hparams = make_train_parser()
    print(hparams)
    print('cuda is available', torch.cuda.is_available())
    if os.path.isdir(os.path.join(hparams.result_path, hparams.exp)) == False:
        os.makedirs(os.path.join(hparams.result_path, hparams.exp))
    train(hparams)