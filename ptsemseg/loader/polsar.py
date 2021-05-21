'''
Author: Shuailin Chen
Created Date: 2021-03-05
Last Modified: 2021-05-21
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torchvision import transforms
import numpy as np
import cv2
from matplotlib import pyplot as plt 

from mylib import polSAR_utils as psr
from mylib import labelme_utils as lbm
from mylib import file_utils as fu
from ptsemseg.augmentations.augmentations import *


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
    '''
    
    def __init__(self, 
                file_root,        
                split_root,
                split = "train",
                augments=None,
                data_format = 'Hoekman',
                norm=False
                ):
        super().__init__()
        self.file_root = file_root
        self.augments = augments
        self.data_format = data_format
        self.norm = norm
        
        if 'GF3' in file_root:
            self.sensor = 'GF3'
        elif 'RS2' in file_root:
            self.sensor = 'RS2'
        else:
            raise IOError('don\' know the sensor type')
        
        print(f'split: {split}\n\tfile root: {file_root}\n\taugment: {augments}\n\tdata format: {data_format}\n\tnorm: {norm}')

        # read all files' path
        self.files_path = fu.read_file_as_list(osp.join(split_root, split))

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

        else:
            raise NotImplementedError

        return file

    def __getitem__(self, index):
        img = self.get_file_data(index)
        
        # save_dir = './tmp'
        # cv2.imwrite(osp.join(save_dir, 'p_fila_a.png'), (file_a.permute(1,2,0).numpy()*255).astype(np.uint8))
        # cv2.imwrite(osp.join(save_dir, 'p_fila_b.png'), (file_b.permute(1,2,0).numpy()*255).astype(np.uint8))    
        # cv2.imwrite(osp.join(save_dir, 'p_mask.png'), (mask.numpy()*255).astype(np.uint8))
        # cv2.imwrite(osp.join(save_dir, 'p_label.png'), (label.numpy()*255).astype(np.uint8))

        if self.augments:
            img, _ = self.augments(img, img)
        return file_a, file_b, label, mask

        # cv2.imwrite(osp.join(save_dir, 'a_fila_a.png'), (file_a.permute(1,2,0).numpy()*255).astype(np.uint8))
        # cv2.imwrite(osp.join(save_dir, 'a_fila_b.png'), (file_b.permute(1,2,0).numpy()*255).astype(np.uint8))    
        # cv2.imwrite(osp.join(save_dir, 'a_mask.png'), (mask.numpy()*255).astype(np.uint8))
        # cv2.imwrite(osp.join(save_dir, 'a_label.png'), (label.numpy()*255).astype(np.uint8))
    
    def rand_pool(self, img, mask=None):
        ''' Random pooling 
        Args:
            img (ndarray): image file, in shape of [height, weight, channel],
                whose shape should be divisible by 2
            mask (list): subsample mask, if None, this func generates one. 
                Default: None
        Returns:
            sub_img (list): two subsampled images
            mask (list): subsample mask
        '''

        # generate mask if None
        if mask is None:
            height, weight, channel = img.shape
            pool_enc = np.random.randint(0, 6, size=(height//2, weight//2))
            pool_dec_1 = (pool_enc-1) // 2
            pool_dec_1[pool_dec_1<0] = 0        # because -1//2 = 1
            pool_dec_2 = (pool_enc)%3 + pool_dec_1 + 1
            mask = [pool_dec_1, pool_dec_2]
            shuffle(mask)

        # apply subsample mask
        img = img.reshape(height//2, 2, weight//2, 2, -1)
        img = img.transpose(0, 2, 1, 3, 4).reshape(height//2, weight//2, 4, -1)
        sub_img = []
        sub_img.append(img[:, :, mask[0], :])
        sub_img.append(img[:, :, mask[1], :])

        return sub_img, mask


if __name__=='__main__':
    save_dir = './tmp'
    ds = PolSAR(root=r'/data/csl/SAR_CD/GF3', split='train', data_format='pauli', augments=Compose(RandomHorizontalFlip(0.5)))
    idx = 73
    ds.__getitem__(idx)

    file_a, file_b = ds.get_files_data(idx)
    label, mask = ds.get_label_and_mask(idx)
    ds.statistics()
    cv2.imwrite(osp.join(save_dir, 'fila_a.png'), (file_a.permute(1,2,0).numpy()*255).astype(np.uint8))
    cv2.imwrite(osp.join(save_dir, 'fila_b.png'), (file_b.permute(1,2,0).numpy()*255).astype(np.uint8))    
    cv2.imwrite(osp.join(save_dir, 'mask.png'), (mask.numpy()*255).astype(np.uint8))
    cv2.imwrite(osp.join(save_dir, 'label.png'), (label.numpy()*255).astype(np.uint8))
    print('done')

