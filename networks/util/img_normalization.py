from __future__ import division, print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImgNormalizerForResnet(nn.Module):
    def __init__(self):
        super(ImgNormalizerForResnet, self).__init__()
        IMG_NORM_MEAN = [0.485, 0.456, 0.406]
        IMG_NORM_STD = [0.229, 0.224, 0.225]
        img_mean = np.array(IMG_NORM_MEAN, dtype=np.float32).reshape((1, -1, 1, 1))
        img_std = np.array(IMG_NORM_STD, dtype=np.float32).reshape((1, -1, 1, 1))
        self.register_buffer('img_mean', torch.from_numpy(img_mean))
        self.register_buffer('img_std', torch.from_numpy(img_std))

    def forward(self, imgs):
        imgs_ = F.interpolate(imgs, [224, 224], mode='bilinear')
        imgs_ = (imgs_ - self.img_mean) / self.img_std
        return imgs_