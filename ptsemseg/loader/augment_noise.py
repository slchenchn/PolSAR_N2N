'''
Author: Shuailin Chen
Created Date: 2021-06-30
Last Modified: 2021-06-30
	content: noise generator, adapted from "https://github.com/TaoHuang2018/Neighbor2Neighbor/blob/main/training_script.md"
'''
import torch
import numpy as np

from .rand_pool import get_generator


operation_seed_counter = 0


class AugmentNoise(object):
    ''' Copyied from "https://github.com/TaoHuang2018/Neighbor2Neighbor/blob/main/training_script.md"
    '''
    def __init__(self, style):
        if style.startswith('gauss'):
            self.params = [float(p) / 255.0 for p in style.replace('gauss', '', 1).split('_')]
            if len(self.params) == 1:
                self.style = "gauss_fix"
            elif len(self.params) == 2:
                self.style = "gauss_range"
        elif style.startswith('poisson'):  
            self.params = [float(p) for p in style.replace('poisson', '', 1).split('_')]
            if len(self.params) == 1:
                self.style = "poisson_fix"
            elif len(self.params) == 2:
                self.style = "poisson_range"

    def add_train_noise(self, x):
        shape = x.shape
        if self.style == "gauss_fix":
            std = self.params[0]
            std = std * torch.ones((shape[0], 1, 1, 1), device=x.device)
            noise = torch.cuda.FloatTensor(shape, device=x.device)
            torch.normal(mean=0.0, std=std, generator=get_generator(), out=noise)
            return x + noise
        elif self.style == "gauss_range":
            min_std, max_std = self.params
            std = torch.rand(size=(shape[0], 1, 1, 1), device=x.device) * (max_std - min_std) + min_std
            noise = torch.cuda.FloatTensor(shape, device=x.device)
            torch.normal(mean=0, std=std, generator=get_generator(), out=noise)
            return x + noise
        elif self.style == "poisson_fix":
            lam = self.params[0]
            lam = lam * torch.ones((shape[0], 1, 1, 1), device=x.device)
            noised = torch.poisson(lam * x, generator=get_generator()) / lam
            return noised
        elif self.style == "poisson_range":
            min_lam, max_lam = self.params
            lam = torch.rand(size=(shape[0], 1, 1, 1), device=x.device) * (max_lam - min_lam) + min_lam
            noised = torch.poisson(lam * x, generator=get_generator()) / lam
            return noised

    def add_valid_noise(self, x):
        shape = x.shape
        if self.style == "gauss_fix":
            std = self.params[0]
            return np.array(x + np.random.normal(size=shape) * std, dtype=np.float32)
        elif self.style == "gauss_range":
            min_std, max_std = self.params
            std = np.random.uniform(low=min_std, high=max_std, size=(1, 1, 1))
            return np.array(x + np.random.normal(size=shape) * std, dtype=np.float32)
        elif self.style == "poisson_fix":
            lam = self.params[0]
            return np.array(np.random.poisson(lam * x) / lam, dtype=np.float32)
        elif self.style == "poisson_range":
            min_lam, max_lam = self.params
            lam = np.random.uniform(low=min_lam, high=max_lam, size=(1, 1, 1))
            return np.array(np.random.poisson(lam * x) / lam, dtype=np.float32)


class MyNoiseAdder(AugmentNoise):
    ''' wrapper of noise adders '''
    
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def add_train_noise(self, x):
        return x, super().add_train_noise(x)

    def add_valid_noise(self, x):
        return x, super().add_valid_noise(x)