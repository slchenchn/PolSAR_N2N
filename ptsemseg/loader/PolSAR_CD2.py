'''
Author: Shuailin Chen
Created Date: 2021-03-05
Last Modified: 2021-04-01
	content: 
'''
import argparse
from operator import imod, index, truediv
from re import search
import shutil
from numpy.lib.npyio import save
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torchvision import transforms
import os
import os.path as osp
import sys
from typing import Union
# sys.path.append(osp.abspath(os.getcwd()))   #很奇怪，必须用绝对路径
# print('sys path: ', sys.path)
from mylib import polSAR_utils as psr
from mylib import labelme_utils as lbm
import re
from matplotlib import pyplot as plt 
import numpy as np
import cv2
from ptsemseg.augmentations.CD_augments import *

class PolSAR_CD_base(data.Dataset):
    ''' the base class of the PolSAR change detection dataloaders '''
    def __init__(self, root,        
        split="train",
        # img_size=512,
        augments=None,
        to_tensor = True,
        data_format = 'pauli',
        use_perc = 1.0,
        norm=True
        # data_type = 'pauli'
        ):
        super().__init__()
        self.root = root
        self.split = split
        self.data_format = data_format
        self.augments = augments
        self.to_tensor = to_tensor
        self.sensor = root[-3:]
        self.norm = norm
        # self.data_type = data_format
        print(f'split: {split}\n\troot: {root}\n\taugment: {augments}\n\tto tensor: {to_tensor}\n\tdata format: {data_format}')

        # read change label images' path
        self.labels_path = []
        if self.split=='train2':
            # samples the used data
            all_labels_path = [[]]
            with open(osp.join(root, split+'.txt')) as f:
                for line in f:
                    if '\n'==line:
                        all_labels_path.append([])
                    else:
                        all_labels_path[-1].append(osp.join(root, line.strip()))
            for label_class in all_labels_path:
                self.labels_path += label_class[:int(np.round(len(label_class)*use_perc))]
                # raise NotImplementedError
        
        else:
            with open(osp.join(root, split+'.txt')) as f:
                for line in f:
                    self.labels_path.append(osp.join(root, line.strip()))

        # for super_dir, _, files in os.walk(self.root):
        #     for file in files:
        #         if '-change.png' in file:
        #             self.labels_path.append(osp.join(super_dir, file))

        # transform
        if self.to_tensor:
            self.tf = transforms.ToTensor()

    def __len__(self):
        return len(self.labels_path)

    def get_label_and_mask(self, index:int):
        ''' generate label and its mask, in torch.tensor forat '''
        # print('index: ', index)
        label_path = self.labels_path[index]
        label = lbm.read_change_label_png(label_path)-1
        mask = label<2      # 1 表示存在有效标记，0表示没有标记
        label[~mask] = 0    # 1 表示存在变化，0表示没有变化或没有标记数据
        # cv2.imwrite('tmp/mask.png', (mask*255).astype(np.uint8))
        # cv2.imwrite('tmp/label.png', (label*255).astype(np.uint8))
        return torch.from_numpy(label).long(), torch.from_numpy(mask)

    def get_files_data(self, index):
        ''' read file a and file b data, in torch.tensor format
        if the data is images, then it is normed into [0, 1], 
        '''
        label_path = self.labels_path[index]
        # get the file path
        label_dir = osp.split(label_path)[0]
        # two date time display format
        re_exp = r'20\d{6}'
        [a_time, b_time] = re.findall(re_exp, label_path)

        files_path = []
        if 's2' in self.data_format:
            files_path.append(osp.join(label_dir.replace('label', 'data'), a_time, 's2'))
            files_path.append(osp.join(label_dir.replace('label', 'data'), b_time, 's2'))
        else:
            files_path.append(osp.join(label_dir.replace('label', 'data'), a_time, 'C3'))
            files_path.append(osp.join(label_dir.replace('label', 'data'), b_time, 'C3'))

        # get the file data
        if self.sensor=='GF3':
            slice_idx = re.search(r'-\d{3}-', label_path)
        elif self.sensor=='RS2':
            slice_idx = re.search(r'-\d{4}-', label_path)
        if slice_idx is None:
            raise ValueError('can not find the wave code')
        slice_idx = int(slice_idx.group()[1:-1])

        files = []
        if self.data_format in ('save_space', 'complex_vector_9', 'complex_vector_6'):
            for ii in range(2):
                psr_data = psr.read_c3(osp.join(files_path[ii], str(slice_idx)), out=self.data_format)
                if self.norm:
                    mean = np.load(osp.join(files_path[ii], self.data_format+'_mean.npy'))
                    std = np.load(osp.join(files_path[ii], self.data_format+'_std.npy'))
                    _, _, psr_data = psr.norm_3_sigma(psr_data, mean, std)
                files.append(torch.from_numpy(psr_data).type(torch.complex64))

        # polar coordinate form, i.e. magnitude and angle
        elif 'polar' in self.data_format:
            if 'c3' in self.data_format:
                # complex vector 6
                for ii in range(2):
                    c6 = psr.read_c3(osp.join(files_path[ii], str(slice_idx)), out='complex_vector_6').astype(np.complex64)
                    if self.norm:
                        mean = np.load(osp.join(files_path[ii], 'complex_vector_6'+'_mean.npy'))
                        std = np.load(osp.join(files_path[ii], 'complex_vector_6'+'_std.npy'))
                        _, _, c6 = psr.norm_3_sigma(c6, mean, std)
                    abs = np.expand_dims(np.abs(c6), axis=0)
                    agl = np.expand_dims(np.angle(c6), axis=0)
                    polar = torch.cat((torch.from_numpy(agl),torch.from_numpy(abs)), dim=0)
                    if '4D' in self.data_format:
                        x = torch.from_numpy(c6.real.astype(np.float32))
                        y = torch.from_numpy(c6.imag.astype(np.float32))
                        polar = torch.cat((polar, x, y), dim=0)
                    files.append(polar)

            elif 's2' in self.data_format:
                # s2 matrix
                for ii in range(2):
                    s2 = psr.read_s2(osp.join(files_path[ii], str(slice_idx))).astype(np.complex64)
                    if self.norm:
                        mean = np.load(osp.join(files_path[ii], 's2_abs_mean.npy'))
                        std = np.load(osp.join(files_path[ii], 's2_abs_std.npy'))
                        _, _, s2 = psr.norm_3_sigma(s2, mean, std, type='abs')
                    abs = np.expand_dims(np.abs(s2), axis=0)
                    agl = np.expand_dims(np.angle(s2), axis=0)
                    polar = torch.cat((torch.from_numpy(agl),torch.from_numpy(abs)), dim=0)
                    if '4D' in self.data_format:
                        x = torch.from_numpy(s2.real.astype(np.float32)).unsqueeze(dim=0)
                        y = torch.from_numpy(s2.imag.astype(np.float32)).unsqueeze(dim=0)
                        polar = torch.cat((polar, x, y), dim=0)
                    files.append(polar)

        elif self.data_format == 's2':
            for ii in range(2):
                psr_data = psr.read_s2(osp.join(files_path[ii], str(slice_idx)))
                if self.norm:
                    mean = np.load(osp.join(files_path[ii], 's2_abs_mean.npy'))
                    std = np.load(osp.join(files_path[ii], 's2_abs_std.npy'))
                    _, _, psr_data = psr.norm_3_sigma(psr_data, mean, std, type='abs')
                files.append(torch.from_numpy(psr_data).type(torch.complex64))

        elif self.data_format=='pauli':
            for ii in range(2):
                files.append(psr.read_bmp(osp.join(files_path[ii], str(slice_idx))))
            if self.to_tensor:
                for ii in range(2):
                    files[ii] = self.tf(files[ii])
            else:
                for ii in range(2):
                    files[ii] = files[ii].permute(2, 0, 1)
        elif self.data_format=='hoekman':
            for ii in range(2):
                files.append(torch.from_numpy(np.load(osp.join(files_path[ii], str(slice_idx), 'normed.npy').replace('C3', 'Hoekman'))))
        else:
            raise NotImplementedError
        return files

    def __getitem__(self, index):
        label, mask = self.get_label_and_mask(index)
        file_a, file_b = self.get_files_data(index)
        
        # save_dir = './tmp'
        # cv2.imwrite(osp.join(save_dir, 'p_fila_a.png'), (file_a.permute(1,2,0).numpy()*255).astype(np.uint8))
        # cv2.imwrite(osp.join(save_dir, 'p_fila_b.png'), (file_b.permute(1,2,0).numpy()*255).astype(np.uint8))    
        # cv2.imwrite(osp.join(save_dir, 'p_mask.png'), (mask.numpy()*255).astype(np.uint8))
        # cv2.imwrite(osp.join(save_dir, 'p_label.png'), (label.numpy()*255).astype(np.uint8))

        if self.augments:
            file_a, file_b, label, mask = self.augments(file_a, file_b, label, mask)
        return file_a, file_b, label, mask

        # cv2.imwrite(osp.join(save_dir, 'a_fila_a.png'), (file_a.permute(1,2,0).numpy()*255).astype(np.uint8))
        # cv2.imwrite(osp.join(save_dir, 'a_fila_b.png'), (file_b.permute(1,2,0).numpy()*255).astype(np.uint8))    
        # cv2.imwrite(osp.join(save_dir, 'a_mask.png'), (mask.numpy()*255).astype(np.uint8))
        # cv2.imwrite(osp.join(save_dir, 'a_label.png'), (label.numpy()*255).astype(np.uint8))
        
    def statistics(self):
        ''' calculate the statistics of the labels'''
        # absolute value
        cnt_change = 0
        cnt_unchange = 0
        cnt_unlabeled = 0
        for label_path in self.labels_path:
            label = lbm.read_change_label_png(label_path)
            cnt_unlabeled += np.count_nonzero(label==0)
            cnt_unchange += np.count_nonzero(label==1)
            cnt_change += np.count_nonzero(label==2)
            
        # percentage
        cnt_all = cnt_change + cnt_unchange + cnt_unlabeled
        if cnt_all!=len(self)*512**2:
            print('cnt_all wrong, cnt_all=', cnt_all, ', actually it should be', len(self)*512**2)
        pec_change = cnt_change / cnt_all
        pec_unchange = cnt_unchange / cnt_all
        per_unlabeled = cnt_unlabeled / cnt_all

        # prints
        print(f'number of samples: {self.__len__()}')
        print(f'changed: {cnt_change}, {pec_change*100}%')
        print(f'unchanged: {cnt_unchange}, {pec_unchange*100}%')
        print(f'unlabeled: {cnt_unlabeled}, {per_unlabeled*100}%')
        print('unchanged / changed:', cnt_unchange/cnt_change)


if __name__=='__main__':
    save_dir = './tmp'
    ds = PolSAR_CD_base(root=r'/data/csl/SAR_CD/GF3', split='train', data_format='pauli', augments=Compose(RandomHorizontalFlip(0.5)))
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

