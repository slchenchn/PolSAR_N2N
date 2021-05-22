'''
Author: Shuailin Chen
Created Date: 2020-11-27
Last Modified: 2021-05-22
	content: 
'''
import math
import glob
import natsort
import os.path as osp
import os
import re
import random

import torch
import numpy as np

from mylib import types
from mylib import file_utils as fu

def get_work_dir(run_id:str):
    ''' get the full work dir (name+index) give the name '''
    all_runs = glob.glob(run_id+'*')
    all_runs = [run for run in all_runs if re.match(run_id+'_*\d+', run)]
    all_runs = natsort.natsorted(all_runs)
    if all_runs:
        run_id_cnt = re.findall('_\d+', all_runs[-1])
        run_id_cnt = int(run_id_cnt[-1][1:])
        run_id  = run_id + '_' + str(run_id_cnt+1)
    else:
        run_id = run_id + '_0'
    os.mkdir((run_id))
    return run_id


class RandPool():
    ''' random 2x2 pooling '''

    def __init__(self, img) -> None:
        ''' encode and decode to get indices
        Args:
            img (Tensor): [NxCxHxW] shape of tensor
        '''
        
        # enc:      0   1   2   3   4   5  
        # dec_1:    0   0   0   1   1   2
        # dec_2:    1   2   3   2   3   3
        n, h, w, _ = img.shape
        pool_enc = torch.randint(0, 6, size=(n, h//2, w//2))
        self.pool_dec_1 = (pool_enc-1) // 2
        self.pool_dec_1[self.pool_dec_1<0] = 0
        self.pool_dec_2 = pool_enc%3 + self.pool_dec_1 + 1
        self.pool_dec_2[pool_enc==5] = 3


    def apply_mask(self, img):
        ''' apply created mask to a image '''

        n, _, h, w = img.shape
        assert self.pool_dec_1.shape
        # retrive sub image
        h_idx = np.arange(h//2).repeat(w//2)
        w_idx = np.tile(np.arange(w//2), h//2)

        n, h, w, c = img.shape
        assert
        img_ = img.reshape(h//2, 2, w//2, 2, c).transpose(0, 2, 1, 3, 4).reshape(h//2, w//2, 4, c)
        sub_img = []
        sub_img.append(img_[h_idx, w_idx, self.pool_dec_1.flatten(), :].reshape(h//2, w//2, c))
        sub_img.append(img_[h_idx, w_idx, self.pool_dec_2.flatten(), :].reshape(h//2, w//2, c))
        return sub_img


def split_train_val_test(src_path, dst_path, data_format, train_ratio=0.8):
    ''' Random split the training, validation, test set using specific ratio of number of training samples

    Args:
        src_path (str): PolSAR file path
        dst_path (str): path to write split file
        data_format (str): 'Hoekman' or 's2' or 'C3
        train_ratio (float): ratio of number of training samples. Default: 0.8
    '''

    # collect all files
    all_filenames = dict()
    for location in os.listdir(src_path):
        all_filenames[location] = []
        for root, dirs, _ in os.walk(osp.join(src_path, location)):
            if root.endswith(data_format):
                for dir in dirs:
                    all_filenames[location].append(osp.join(root, dir))
    
    # split
    val_ratio = (1-train_ratio)/2
    test_ratio = (1-train_ratio)/2
    val_split = []
    test_split =  []
    train_split = []
    num_all_files = 0
    for k, v in all_filenames.items():
        num_all_files += len(v)
        num_val = round(len(v)*val_ratio)
        num_test = round(len(v)*test_ratio)
        val_test_idx = random.sample(range(len(v)), num_val+num_test)
        val_test, train = types.list_pop(v, val_test_idx)
        train_split += train
        val_split += val_test[:len(val_test)//2]
        test_split += val_test[len(val_test)//2:]

    print(f'num of all files: {num_all_files}')
    print('num of train split: ', len(train_split))
    print('num of val split: ', len(val_split))
    print('num of test split: ', len(test_split))

    assert len(train_split) + len(val_split) + len(test_split) == num_all_files
    # save to file
    dst_path = osp.join(dst_path, data_format, str(train_ratio))
    fu.mkdir_if_not_exist(dst_path)

    with open(osp.join(dst_path, 'val.txt'), 'w') as f:
        for item in val_split:
            f.write(f'{item}\n')

    with open(osp.join(dst_path, 'train.txt'), 'w') as f:
        for item in train_split:
            f.write(f'{item}\n')
            
    with open(osp.join(dst_path, 'test.txt'), 'w') as f:
        for item in test_split:
            f.write(f'{item}\n')

        
if __name__=='__main__':
    src_path = r'data/GF3/data'
    dst_path = r'data/GF3/denoise/split'
    split_train_val_test(src_path, dst_path, 'Hoekman', train_ratio=0.9)