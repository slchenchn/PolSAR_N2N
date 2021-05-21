'''
Author: Shuailin Chen
Created Date: 2021-03-11
Last Modified: 2021-04-11
	content: 
'''
''' 适用于变化检测的数据增广方式 '''

import math
import numbers
import random

import numpy as np
from numpy import ndarray
from PIL import Image, ImageOps
import torchvision.transforms as tt
import torchvision.transforms.functional as tf
import torchvision.transforms.functional_tensor as F_t
import torch
from torch import nn
import torch.nn.functional as F
from scipy import ndimage
from scipy import misc
import cv2

import mylib.polSAR_utils as psr

class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, file_a, file_b, label, mask):
        # file_a = Image.fromarray(file_a, mode="F")
        # file_b = Image.fromarray(file_b, mode="F")
        # label = Image.fromarray(label, mode="L")
        # mask = Image.fromarray(mask, mode="L")

        for a in self.augmentations:
            file_a, file_b, label, mask = a(file_a, file_b, label, mask)

        # file_a, file_b, label, mask = torch.from_numpy(file_a), torch.from_numpy(file_b), torch.from_numpy(label), torch.from_numpy(mask)

        return file_a, file_b, label, mask


class BoxcarSmooth(object):
    ''' boxcar smoothings '''
    def __init__(self, kernel_size=3, p=0.5) -> None:
        super().__init__()
        self.p = p
        self.kernel_size = kernel_size

    def __call__(self, file_a, file_b, label, mask):
        # assume the file has [channel, height, width] dimension
        if random.random()<self.p:
            ff_a_r = torch.from_numpy(ndimage.uniform_filter(file_a.real, (0, self.kernel_size, self.kernel_size), mode='mirror'))
            ff_a_i = torch.from_numpy(ndimage.uniform_filter(file_a.imag, (0, self.kernel_size, self.kernel_size), mode='mirror'))
            ff_b_r = torch.from_numpy(ndimage.uniform_filter(file_b.real, (0, self.kernel_size, self.kernel_size), mode='mirror'))
            ff_b_i = torch.from_numpy(ndimage.uniform_filter(file_b.imag, (0, self.kernel_size, self.kernel_size), mode='mirror'))
            return ff_a_r+1j*ff_a_i, ff_b_r+1j*ff_b_i, label, mask
        else:
            return file_a, file_b, label, mask


class RandomHorizontalFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, file_a, file_b, label, mask):
        if random.random() < self.p:
            # print('augment:  RandomHorizontallyFlip')
            return (
                torch.flip(file_a, (-1,)),
                torch.flip(file_b, (-1,)),
                torch.flip(label, (-1,)),
                torch.flip(mask, (-1,)),
                # file_a.transpose(Image.FLIP_LEFT_RIGHT),
                # file_b.transpose(Image.FLIP_LEFT_RIGHT),
                # label.transpose(Image.FLIP_LEFT_RIGHT),
                # mask.transpose(Image.FLIP_LEFT_RIGHT),
            )
        return file_a, file_b, label, mask

        
class RandomVerticalFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, file_a, file_b, label, mask):
        if random.random() < self.p:
            # print('augment:  RandomHorizontallyFlip')
            return (
                torch.flip(file_a, (-2,)),
                torch.flip(file_b, (-2,)),
                torch.flip(label, (-2,)),
                torch.flip(mask, (-2,)),
                # file_a.transpose(Image.FLIP_LEFT_RIGHT),
                # file_b.transpose(Image.FLIP_LEFT_RIGHT),
                # label.transpose(Image.FLIP_LEFT_RIGHT),
                # mask.transpose(Image.FLIP_LEFT_RIGHT),
            )
        return file_a, file_b, label, mask


class RandomRotation(object):
    def __init__(self, degrees, 
    interpolation=0, # 0是最近邻，2是双线性
    expand=False, fill=None) -> None:
        super().__init__()
        self.degrees = _setup_angle(degrees)
        # self.center = center
        self.resample = interpolation
        self.expand = expand
        self.fill = fill

    def __call__(self, file_a, file_b, label, mask):
        angle = float(torch.empty(1).uniform_(float(self.degrees[0]), float(self.degrees[1])).item())
        # print('angle: ', angle)
        center_f = [0.0, 0.0]
        matrix = tf._get_inverse_affine_matrix(center_f, -angle, [0.0, 0.0], 1.0, [0.0, 0.0])
        return (
            F_t.rotate(file_a, matrix=matrix, resample=self.resample, expand=self.expand, fill=self.fill),
            F_t.rotate(file_b, matrix=matrix, resample=self.resample, expand=self.expand, fill=self.fill),
            F_t.rotate(label.unsqueeze(0), matrix=matrix, resample=self.resample, expand=self.expand, fill=self.fill).squeeze(0),
            F_t.rotate(mask.unsqueeze(0), matrix=matrix, resample=self.resample, expand=self.expand, fill=self.fill).squeeze(0)
        )
        # catted = torch.cat(file_a, file_b, label, mask, dim=)

def _setup_angle(x):
    if isinstance(x, numbers.Number):
        x = [-x, x]

    return [float(d) for d in x]


class WishartNoise(object):
    ''' Generate Wishart distribution noise according to specified number of looks '''
    def __init__(self, ENL) -> None:
        super().__init__()
        self.ENL = ENL
    
    def __call__(self, file_a, file_b, label, mask) :
        # assume the file has [channel, height, width] dimension
        # file_a and file_b should have ENL>3
        c, h, w = file_a.shape
        if c==6:
            ori_format = 'complex_vector_6'
        elif c==9:
            ori_format = 'complex_vector_9'
        else:
            raise NotImplementedError

        file_a = psr.as_format(file_a, 'complex_vector_9')
        file_b = psr.as_format(file_b, 'complex_vector_9')
        file_a = psr.wishart_noise(file_a.reshape(3, 3, -1), ENL=self.ENL).reshape(9, h, w)
        file_b = psr.wishart_noise(file_b.reshape(3, 3, -1), ENL=self.ENL).reshape(9, h, w)
        file_a = psr.as_format(file_a, ori_format)
        file_b = psr.as_format(file_b, ori_format)

        return torch.from_numpy(file_a), torch.from_numpy(file_b), label, mask


if __name__=='__main__':
    ''' test WishartNoise() '''
    path_a = r'data/GF3/data/E115_N39_中国河北/降轨/1/20161209'
    path_b = r'data/GF3/data/E115_N39_中国河北/降轨/1/20170306'
    file_a = psr.read_c3(path_a)
    file_b = psr.read_c3(path_b)

    # boxcar smoothing
    b = BoxcarSmooth(3)
    file_a, file_b, _, _ = b(file_a, file_b, None, None)
    ba = psr.rgb_by_c3(file_a)
    bb = psr.rgb_by_c3(file_b)
    cv2.imwrite('/home/csl/code/PolSAR_CD/tmp/ba.png', cv2.cvtColor((255*ba).astype(np.uint8), cv2.COLOR_RGB2BGR))
    cv2.imwrite('/home/csl/code/PolSAR_CD/tmp/bb.png', cv2.cvtColor((255*bb).astype(np.uint8), cv2.COLOR_RGB2BGR))

    # generate Wishart noise
    w = WishartNoise(3)
    file_a, file_b, _, _ = w(file_a, file_b, None, None)
    pa = psr.rgb_by_c3(file_a)
    pb = psr.rgb_by_c3(file_b)
    cv2.imwrite('/home/csl/code/PolSAR_CD/tmp/wa.png', cv2.cvtColor((255*pa).astype(np.uint8), cv2.COLOR_RGB2BGR))
    cv2.imwrite('/home/csl/code/PolSAR_CD/tmp/wb.png', cv2.cvtColor((255*pb).astype(np.uint8), cv2.COLOR_RGB2BGR))
    



    ''' test BoxcarSmooth() '''
    # a = torch.arange(32, dtype=float).reshape(2, 4, 4)
    # f = BoxcarSmooth(3)
    # b = f(a)
    # print(a)
    # print(b)
    # print('done')