from opts import make_test_parser
import torch
from tqdm import tqdm
import os
import imageio
from dataset.VCTK import VCTKData
from torch.utils.data import DataLoader
from models.AudioUNet import AudioUNet
from torch.nn import MSELoss
from metric.metrics import AvgPSNR

def test(hparams):
    # setting log infomation
    log_save_path = os.path.join(hparams.log_path, hparams.exp)
    if os.path.isdir(log_save_path) ==False:
        os.mkdir(log_save_path)
    log_save_path=log_save_path + '/test_log.txt'
    
    # setting dataset
    test_data = VCTKData(hparams, 'test')
    test_loader = DataLoader(test_data, batch_size = hparams.batch_size, shuffle = False, pin_memory = True)
    # load checkpoint
    checkpoint = torch.load(hparams.ckpt)
    model = AudioUNet(
        checkpoint['num_filters'],
        checkpoint['num_filter_sizes'],
        checkpoint['up_scale'],
        checkpoint['num_blocks']
        )
    model.load_state_dict(checkpoint['model_state_dict'])

    criterion = MSELoss()

    model.eval()
    # test
    model.eval()
    total_loss = 0
    predict_hr_list = []
    gt_hr_list = []

    with torch.no_grad():
        for data in tqdm(test_loader):
            batch_test_lr = data['lr'].cuda()
            batch_test_hr = data['hr'].cuda()
            predict_test_hr = model(batch_test_lr)
            # compute loss
            loss = criterion(predict_test_hr, batch_test_hr)
            total_loss += loss.detach().item()
            # save predict high resolution audio clip and ground truth high resolution audio clip
            predict_hr_list.append(predict_test_hr.detach().cpu().numpy())
            gt_hr_list.append(batch_test_hr.detach().cpu().numpy())

    # write info to log
    with open( log_save_path,'w') as fh:
        fh.seek(0)
        fh.write(f'MSE loss : {total_loss} \n')
        fh.write(f'AVG PSNR : {AvgPSNR(predict_hr_list, gt_hr_list)} \n')
        
    
    # TODO: save the spectrogram
    # img=(all_predict_rgb.numpy()*255).astype(np.uint8).reshape(test_data.img_size)
    # imageio.imwrite(f'results/{hparams.exp}/test.png', img)
if __name__=='__main__':
    hparams = make_test_parser()
    test(hparams)
    