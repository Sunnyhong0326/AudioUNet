import librosa
import numpy as np
import matplotlib.pyplot as plt

from models.AudioUNet import AudioUNet
import torch

from scipy import interpolate
from scipy.signal import decimate

#origin paper provide spectrugram
def get_spectrum(x, n_fft = 2048):
  S = librosa.stft(x, n_fft = n_fft)
  S = np.log1p(np.abs(S))
  return S

def plot_loss_curve(training_result_dir, metrics, epoch):
    # Plot the loss curve against epoch
    plt.figure(dpi = 100)
    plt.title('Loss (MSE)')
    plt.plot(range(epoch + 1), metrics['train_loss'], label='Train')
    plt.plot(range(epoch + 1), metrics['valid_loss'], label='Valid')
    plt.legend()
    plt.show()
    plt.savefig(str(training_result_dir / 'metrics.jpg'))
    plt.close()

def spline_up(x_lr, r):
  x_lr = x_lr.flatten()
  x_hr_len = len(x_lr) * r
  x_sp = np.zeros(x_hr_len)

  i_lr = np.arange(x_hr_len, step=r)
  i_hr = np.arange(x_hr_len)

  f = interpolate.splrep(i_lr, x_lr)
  x_sp = interpolate.splev(i_hr, f)

  return x_sp

def up_sample_wav_12_to_48(test_audio_path, args):
    '''
    read audio, normalized to -1 and 1 and crop the audio sample points to multiple of args.patch_size
    '''
    # check device if CPU or GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model and hyper parameters
    checkpoint = torch.load(args.ckpt)
    model = AudioUNet(
        checkpoint['num_filters'],
        checkpoint['num_filter_sizes'],
        checkpoint['up_scale'],
        checkpoint['num_blocks']
        )
    patch_size = checkpoint['patch_size']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # preprocess the low resolution audio to be multiple of model patch_size
    audio_lr, _ = librosa.load(test_audio_path, args.in_sr)
    audio_lr  = np.pad(audio_lr, (0, args.patch_size - (audio_lr.shape[0] % patch_size)), 'constant', constant_values=(0,0))
    audio_lr = decimate(audio_lr, 1)
    
    # normalize the amplitude to -1 and 1 and reshape to (1, 1, len(audio))
    x_scale = np.max(np.abs(audio_lr))
    audio_lr = audio_lr / x_scale
    audio_lr = audio_lr.reshape((1, 1, len(audio_lr)))

    # preprocessing
    assert len(audio_lr) == 1
    x_sp = spline_up(audio_lr, args.r)
    x_sp = x_sp[: len(x_sp) - (len(x_sp) % (2**(args.num_blocks+1)))]
    x_sp = x_sp.reshape((1 , 1, len(x_sp)))
    # reshape to (batch, 1, 1892) and change to tensor
    x_sp = x_sp.reshape((int(x_sp.shape[1]/args.patch_size), 1, args.patch_size))
    x_sp = torch.Tensor(x_sp).cuda()

    predict_hr = model(x_sp)
    predict_hr = predict_hr.flatten()
  
    
    return x_sp
