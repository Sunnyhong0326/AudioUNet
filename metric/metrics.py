import numpy as np

def AvgPSNR(predict_hr, gt_hr):
    num_clips = len(predict_hr)
    avg_psnr = 0
    for idx in range(num_clips):
        psnr = PSNR(predict_hr[idx], gt_hr[idx])
        avg_psnr = psnr / num_clips
    return avg_psnr

def PSNR(one_predict_hr, one_gt_hr):
    mse = np.mean((np.array(one_gt_hr, dtype=np.float32) - np.array(one_predict_hr, dtype=np.float32)) ** 2)
    return 20 * np.log10(255 / (np.sqrt(mse)))

def LSD():
    pass