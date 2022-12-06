import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def compare_snr_single_image(img_test, img_true):
    return 20 * np.log10(np.linalg.norm(img_true.flatten()) / np.linalg.norm(img_true.flatten() - img_test.flatten()))


def compare_snr_batch(img_test, img_true):
    Img = img_test.data.cpu().numpy().astype(np.float32).squeeze()
    Iclean = img_true.data.cpu().numpy().astype(np.float32).squeeze()
    SNR = 0
    for i in range(Img.shape[0]):
        SNR += compare_snr_single_image(Img[i,:,:], Iclean[i,:,:])
    return (SNR/Img.shape[0])


def compare_psnr_batch(img_test, img_true, data_range=1.):
    Img = img_test.data.cpu().numpy().astype(np.float32).squeeze()
    Iclean = img_true.data.cpu().numpy().astype(np.float32).squeeze()
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += peak_signal_noise_ratio(Iclean[i,:,:], Img[i,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])

def compare_ssim_batch(img_test, img_true, data_range=1.):
    Img = img_test.data.cpu().numpy().astype(np.float32).squeeze()
    Iclean = img_true.data.cpu().numpy().astype(np.float32).squeeze()
    SSIM = 0
    for i in range(Img.shape[0]):
        SSIM += structural_similarity(Iclean[i,:,:], Img[i,:,:], data_range=data_range)
    return (SSIM/Img.shape[0])