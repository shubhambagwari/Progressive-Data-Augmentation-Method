import torch
import torch.nn.functional as F
import numpy as np
import math
from PIL import Image
import cv2

def gaussian(window_size, sigma):
    """
    Generates a list of Tensor values drawn from a gaussian distribution with standard
    diviation = sigma and sum of all elements = 1.

    Length of list = window_size
    """
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=1):
    # Generate an 1D tensor containing values sampled from a gaussian distribution
    _1d_window = gaussian(window_size=window_size, sigma=1.5).unsqueeze(1)

    # Converting to 2D
    _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)

    window = torch.Tensor(_2d_window.expand(channel, 1, window_size, window_size).contiguous())

    return window


def ssim(img1, img2, val_range, window_size=11, window=None, size_average=True, full=False):
    L = val_range  # L is the dynamic range of the pixel values (255 for 8-bit grayscale images),

    pad = window_size // 2

    try:
        _, channels, height, width = img1.size()
    except:
        channels, height, width = img1.size()

    # if window is not provided, init one
    if window is None:
        real_size = min(window_size, height, width)  # window should be atleast 11x11
        window = create_window(real_size, channel=channels).to(img1.device)

    # calculating the mu parameter (locally) for both images using a gaussian filter
    # calculates the luminosity params
    mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channels)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2

    # now we calculate the sigma square parameter
    # Sigma deals with the contrast component
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu12

    # Some constants for stability
    C1 = (0.01) ** 2  # NOTE: Removed L from here (ref PT implementation)
    C2 = (0.03) ** 2

    contrast_metric = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    contrast_metric = torch.mean(contrast_metric)

    numerator1 = 2 * mu12 + C1
    numerator2 = 2 * sigma12 + C2
    denominator1 = mu1_sq + mu2_sq + C1
    denominator2 = sigma1_sq + sigma2_sq + C2

    ssim_score = (numerator1 * numerator2) / (denominator1 * denominator2)

    if size_average:
        ret = ssim_score.mean()
    else:
        ret = ssim_score.mean(1).mean(1).mean(1)

    if full:
        return ret, contrast_metric

    return ret

def getImg(path):
    img = cv2.imread(path)
    return cv2.resize(img,dsize=(256,256))

if __name__ == "__main__":
    import os
    from multiprocessing.dummy import Pool as ThreadPool

    load_images = lambda x: np.asarray(getImg(x))
    # Helper functions to convert to Tensors
    tensorify = lambda x: torch.Tensor(x.transpose((2, 0, 1))).unsqueeze(0).float().div(255.0)
    scores = []

    # helper function to load images

    gen_dir = './result/'
    sim_dir = './contents/'

    gen_imgs = os.listdir(gen_dir)
    for gen_idx in gen_imgs:
        idx = gen_idx.split('-')[0].split('_')[1].split('.')[0]
        gen_img = load_images(gen_dir + gen_idx)
        sim_img = load_images(sim_dir+'Zumwalt_' + idx +'.jpg')
        _img1 = tensorify(gen_img)
        _img2 = tensorify(sim_img)
        true_vs_false = ssim(_img1, _img2, val_range=255).cpu().numpy()
        scores.append(true_vs_false)
        print("Sim vs Gen Image SSIM Score:", true_vs_false)
        cv2.imshow('gen',gen_img)
        cv2.imshow('sim', sim_img)
        cv2.waitKey()
        print('Average:%f'%(sum(scores)/len(scores)))