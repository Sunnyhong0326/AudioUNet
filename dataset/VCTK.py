import torch
from torch.utils.data import Dataset
import os
from pathlib import Path
import librosa
import numpy as np
import random
import warnings ; warnings.filterwarnings('ignore')


def read_audio(x_audio_path, y_audio_path, in_sr, out_sr, patch_size):
    '''
    1. Load audio
    2. Crop the audio sample points to args.patch_size to match input of the model
    3. normalized to -1 and 1 
    '''
    # load high and low resolution audio
    audio_lr, _ = librosa.load(x_audio_path, sr = in_sr)
    audio_hr, _ = librosa.load(y_audio_path, sr = out_sr)

    # ratio of input sample rate and output sample rate
    ratio = int(out_sr / in_sr)

    # length of low resolution audio
    len_audio_lr = len(audio_lr)
    end_num = len_audio_lr - patch_size - 1
    if end_num <= 0:
        print('Audio too short:', end_num)
        print('Audio path', x_audio_path)
        
    # random select audio clip
    start_idx = random.randint(0, end_num)
    end_idx = start_idx + patch_size

    audio_lr = audio_lr[start_idx: end_idx]
    audio_hr = audio_hr[int(start_idx * ratio): int(end_idx * ratio)]

    # normalize the amplitude to -1 and 1
    x_scale = np.max(np.abs(audio_lr))
    y_scale = np.max(np.abs(audio_hr))
    audio_lr = audio_lr / x_scale
    audio_hr = audio_hr / y_scale

    # shape of audio clip (1, 8192)
    audio_lr = np.expand_dims(audio_lr, axis = 0)
    audio_hr = np.expand_dims(audio_hr, axis = 0)

    # convert to tensor
    audio_lr = torch.tensor(audio_lr)
    audio_hr = torch.tensor(audio_hr)
    return audio_lr, audio_hr

class VCTKData(Dataset):
    def __init__(self, args, split):
        '''
        self.audio_x_path: input low resolution audio directory path
        self.audio_y_path: target high resolution audio directory path
        '''
        audio_dataset_path = os.path.join(args.audio_path, split, split)
        audio_dir = Path(audio_dataset_path)
        x_dir_path = audio_dir / '16k'
        y_dir_path = audio_dir / '48k'

        self.audio_x_path = list()
        self.audio_y_path = list()
        self.in_sr = args.in_sr
        self.out_sr = args.out_sr
        self.patch_size = args.patch_size

        # load input audio path
        for path in x_dir_path.iterdir():
            if path.is_dir():
                for audio in path.iterdir():
                    self.audio_x_path.append(audio)

        # load target audio path
        for path in y_dir_path.iterdir():
            if path.is_dir():
                for audio in path.iterdir():
                    self.audio_y_path.append(audio)
        
    def __len__(self):
        return len(self.audio_x_path)
    
    def __getitem__(self, idx):
        audio_lr, audio_hr = read_audio(
            self.audio_x_path[idx], 
            self.audio_y_path[idx],
            self.in_sr,
            self.out_sr,
            self.patch_size
            )

        return {'lr': audio_lr, 'hr': audio_hr}

