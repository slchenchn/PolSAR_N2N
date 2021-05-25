'''
Author: Shuailin Chen
Created Date: 2021-03-05
Last Modified: 2021-05-25
	content: 
'''
import shutil
import os
import os.path as osp
import sys
from typing import Union
import re
from random import shuffle

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
from mylib import labelme_utils as lbm
from mylib import file_utils as fu
from mylib import types
from mylib import mathlib
from mylib import my_torch_tools as mt
# import ptsemseg.augmentations.augmentations

class PolSAR(data.Dataset):
    ''' PolSAR Neighbor2Neighbor dataloaders 
    
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
        
        # determine sensor type for future purpose
        if 'GF3' in file_root:
            self.sensor = 'GF3'
        elif 'RS2' in file_root:
            self.sensor = 'RS2'
        else:
            raise IOError('don\' know the sensor type')

        # read all files' path
        self.files_path = fu.read_file_as_list(osp.join(split_root, split+'.txt'))
        
        print(f'split: {split}\n\tfile root: {file_root}\n\tsensor: {self.sensor}\n\tdata format: {data_format}\n\tnorm: {norm}\n\tlen: {self.__len__()}')

    def __len__(self):
        return len(self.files_path)

    def get_file_data(self, index):
        ''' Read file data '''

        # get the file data
        folder_path = self.files_path[index]

        if self.data_format in ('save_space', 'complex_vector_9', 'complex_vector_6'):
            raise NotImplementedError
            # psr_data = psr.read_c3(folder_path, out=self.data_format)
            # if self.norm:
            #     mean = np.load(osp.join(files_path[ii], self.data_format+'_mean.npy'))
            #     std = np.load(osp.join(files_path[ii], self.data_format+'_std.npy'))
            #     _, _, psr_data = psr.norm_3_sigma(psr_data, mean, std)
            # file = torch.from_numpy(psr_data).type(torch.complex64)

        # polar coordinate form, i.e. magnitude and angle
        elif 'polar' in self.data_format:
            raise NotImplementedError
            # if 'c3' in self.data_format:
            #     # complex vector 6
            #     for ii in range(2):
            #         c6 = psr.read_c3(osp.join(files_path[ii], str(slice_idx)), out='complex_vector_6').astype(np.complex64)
            #         if self.norm:
            #             mean = np.load(osp.join(files_path[ii], 'complex_vector_6'+'_mean.npy'))
            #             std = np.load(osp.join(files_path[ii], 'complex_vector_6'+'_std.npy'))
            #             _, _, c6 = psr.norm_3_sigma(c6, mean, std)
            #         abs = np.expand_dims(np.abs(c6), axis=0)
            #         agl = np.expand_dims(np.angle(c6), axis=0)
            #         polar = torch.cat((torch.from_numpy(agl),torch.from_numpy(abs)), dim=0)
            #         if '4D' in self.data_format:
            #             x = torch.from_numpy(c6.real.astype(np.float32))
            #             y = torch.from_numpy(c6.imag.astype(np.float32))
            #             polar = torch.cat((polar, x, y), dim=0)
            #         files.append(polar)

            # elif 's2' in self.data_format:
            #     # s2 matrix
            #     for ii in range(2):
            #         s2 = psr.read_s2(osp.join(files_path[ii], str(slice_idx))).astype(np.complex64)
            #         if self.norm:
            #             mean = np.load(osp.join(files_path[ii], 's2_abs_mean.npy'))
            #             std = np.load(osp.join(files_path[ii], 's2_abs_std.npy'))
            #             _, _, s2 = psr.norm_3_sigma(s2, mean, std, type='abs')
            #         abs = np.expand_dims(np.abs(s2), axis=0)
            #         agl = np.expand_dims(np.angle(s2), axis=0)
            #         polar = torch.cat((torch.from_numpy(agl),torch.from_numpy(abs)), dim=0)
            #         if '4D' in self.data_format:
            #             x = torch.from_numpy(s2.real.astype(np.float32)).unsqueeze(dim=0)
            #             y = torch.from_numpy(s2.imag.astype(np.float32)).unsqueeze(dim=0)
            #             polar = torch.cat((polar, x, y), dim=0)
            #         files.append(polar)

        elif self.data_format == 's2':
            raise NotImplementedError
            # for ii in range(2):
            #     psr_data = psr.read_s2(osp.join(files_path[ii], str(slice_idx)))
            #     if self.norm:
            #         mean = np.load(osp.join(files_path[ii], 's2_abs_mean.npy'))
            #         std = np.load(osp.join(files_path[ii], 's2_abs_std.npy'))
            #         _, _, psr_data = psr.norm_3_sigma(psr_data, mean, std, type='abs')
            #     files.append(torch.from_numpy(psr_data).type(torch.complex64))

        elif self.data_format=='Hoekman':
            if self.norm:
                file_path = osp.join(folder_path, 'normed.npy')
            else:
                file_path = osp.join(folder_path, 'unnormed.npy')
            file = np.load(file_path)
            if self.log:
                file = np.log(file) + self.log_compensation

        else:
            raise NotImplementedError

        return file

    def __getitem__(self, index):
        img = torch.from_numpy(self.get_file_data(index))
        
        # cv2.imwrite(osp.join(save_dir, 'p_fila_a.png'), (file_a.permute(1,2,0).numpy()*255).astype(np.uint8))

        if self.augments:
            img = self.augments(img)
        
        # cv2.imwrite(osp.join(save_dir, 'a_fila_a.png'), (file_a.permute(1,2,0).numpy()*255).astype(np.uint8))

        return img


if __name__=='__main__':
        
    h = 100
    img = np.arange(h**2).reshape(h, h, 1)
    img = np.tile(img, (1, 1, 3))
    sub_imgs, mask = PolSAR.rand_pool(img)
    img = np.transpose(img, (2, 0, 1))
    sub_imgs = [np.transpose(sub, (2, 0, 1)) for sub in sub_imgs]

    print(f'before:\n{img}')
    print('\n')
    print(f'after:\n{sub_imgs[0]}\n\n{sub_imgs[1]}\n\nmask:\n{mask}')

    # mask = np.stack(mask).reshape(-1)
    unique, counts = np.unique(mask[0], return_counts=True)
    print(f'hist of mask 0: {dict(zip(unique, counts))}')

    unique, counts = np.unique(mask[1], return_counts=True)
    print(f'hist of mask 1: {dict(zip(unique, counts))}')
    
    print('done')



