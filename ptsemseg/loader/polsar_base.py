'''
Author: Shuailin Chen
Created Date: 2021-07-04
Last Modified: 2021-07-04
	content: 
'''


import torch
from torch import Tensor
from torch.utils import data
import numpy as np

from mylib import polSAR_utils as psr


class PolSARSBase(data.Dataset):
    ''' Base class for PolSAR dataloader '''
    def __init__(self) -> None:
        super().__init__()


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