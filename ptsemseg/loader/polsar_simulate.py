'''
Author: Shuailin Chen
Created Date: 2021-03-05
Last Modified: 2021-05-31
	content: 
'''
import shutil
import os
import os.path as osp
import sys
from typing import Union
import re
from random import shuffle
from glob import glob

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torchvision import transforms
import numpy as np
import cv2
from matplotlib import pyplot as plt 
from scipy import special

from mylib import polSAR_utils as psr
from mylib import simulate
from mylib import labelme_utils as lbm
from mylib import file_utils as fu
from mylib import types
from mylib import mathlib
from mylib import my_torch_tools as mt
# import ptsemseg.augmentations.augmentations

_TMP_PATH = './tmp'

class PolSARSimulate(data.Dataset):
    ''' PolSAR Neighbor2Neighbor dataloaders for simulated data
    
    Args:
        file_root (str): file root path
        split_root (str): split file root path
        split (str): "train" or "val" or "test". Default: "train"
        augments (Compose): compose of augmentations. Default: None
        data_format (str): "Hoekman" or "complex_vector_6" or 
            "complex_vector_9" or 's2'. Default: "Hoekman"
        norm (bool): whether to read normed or unnormed data. Default: False
        log (bool): whether to log transform the data value, only valid for hoekman decomposition. Default: True
        ENL (int): equivalent numober of looks. Default: 1

    '''
    
    def __init__(self, 
                file_root,        
                split_root,
                split = "train",
                augments=None,
                data_format = 'Hoekman',
                norm=False,
                log = True,
                ENL = 1,
                logger = None,
                ):
        super().__init__()
        self.file_root = file_root
        self.augments = augments
        self.data_format = data_format
        self.norm = norm
        self.log = log
        self.ENL = ENL

        # compensate for the mean value of log tranformed intensites
        if log:
            self.log_compensation = np.log(ENL) - special.digamma(ENL)

        # read all files' path
        self.files_path = glob(osp.join(file_root, r'*/*.jpg'))
        
        info = f'split: {split}\n\tfile root: {file_root}\n\tdata format: {data_format}\n\tnorm: {norm}\n\tlen: {self.__len__()}\n\tENL: {ENL}\n\tlog transform:{log}'

        if logger:
            logger.info(info)
        else:
            print(info)

    def __len__(self):
        return len(self.files_path)

    def __getitem__(self, index):

        # read image and simulate Wishart noise
        img = cv2.imread(self.files_path[index]).astype(np.float32)
        # print(f'file path {self.files_path[index]}')
        img = img[:256, :256, :]    # ensure can be divede by 32
        img[img<1] += 1         # avoid zero value


        if self.augments:
            img = self.augments(torch.from_numpy(img))

        if isinstance(img, torch.Tensor):
            img = img.numpy()
        img, noise = simulate.generate_Wishart_noise_from_img(img, self.ENL)        
        

        # pauli_img = psr.rgb_by_c3(img, type='sinclair')
        # pauli_noise = psr.rgb_by_c3(noise, type='sinclair')
        # cv2.imwrite(osp.join(_TMP_PATH, 'pauli_img.jpg'), cv2.cvtColor((255*pauli_img).astype(np.uint8), cv2.COLOR_RGB2BGR))
        # cv2.imwrite(osp.join(_TMP_PATH, 'pauli_noise.jpg'), cv2.cvtColor((255*pauli_noise).astype(np.uint8), cv2.COLOR_RGB2BGR))


        # hoekman decomposition
        img = psr.Hokeman_decomposition(img)
        noise = psr.Hokeman_decomposition(noise)

        # log transform
        if self.log:
            img = np.log(img) + self.log_compensation
            noise = np.log(noise) + self.log_compensation
        
        # to pytorch
        img = torch.from_numpy(img)
        noise = torch.from_numpy(noise)

        return img, noise, self.files_path[index]

    def Hoekman_recover_to_C3(self, h):
        ''' Recover hoekman data to C3 data

        Args:
            h (Tensor): in shape of [batch, channel, height, width]
        
        Returns
            C3 (ndarray): C3 matrix, in shape of [batch, channel, height, width]
        '''

        # to numpy
        if 'cpu' != h.device:
            h = h.cpu()
        if isinstance(h, Tensor):
            h = h.numpy()

        # inverse log transform
        if self.log:
            h = np.exp(h - self.log_compensation)

        # invere hoekman decomposition
        C3 = np.empty(h.shape)
        for ii in range(h.shape[0]):
            C3[ii, ...] = psr.inverse_Hokeman_decomposition(h[ii, ...])

        return C3


if __name__=='__main__':
        
    h = 100
    img = np.arange(h**2).reshape(h, h, 1)
    img = np.tile(img, (1, 1, 3))
    img = np.transpose(img, (2, 0, 1))

    print(f'before:\n{img}')
    print('\n')
    
    print('done')



