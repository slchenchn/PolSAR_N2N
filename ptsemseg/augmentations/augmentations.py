'''
Author: Shuailin Chen
Created Date: 2021-03-11
Last Modified: 2021-05-21
	content: augmentations for this project
'''

import math
import numbers
import random
import numpy as np
import torchvision.transforms as tt
import torchvision.transforms.functional as tf
import torchvision.transforms.functional_tensor as F_t
import torch
from PIL import Image, ImageOps
from torch import nn
import torch.nn.functional as F


class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, file):
        for a in self.augmentations:
            file= a(file)

        return file
 

# class Boxcar_smooth(object):
#     def __init__(self, kernel_size=3) -> None:
#         super().__init__()
#         self.kernel_size = kernel_size
#         self.kernel = torch.ones(kernel_size, kernel_size)/(kernel_size**2)
#         self.padding = int((kernel_size-1)/2)

#     def __call__(self, file):
#         return F.conv2d(file, self.kernel, bias=None, padding=self.padding)


class RandomHorizontalFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, file):
        if random.random() < self.p:
            # print('augment:  RandomHorizontallyFlip')
            return torch.flip(file, (-1,))
            
        return file

        
class RandomVerticalFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, file):
        if random.random() < self.p:
            # print('augment:  RandomHorizontallyFlip')
            return torch.flip(file, (-2,))
            
        return file


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

    def __call__(self, file):
        file = torch.from_numpy(file)
        angle = float(torch.empty(1).uniform_(float(self.degrees[0]), float(self.degrees[1])).item())
        # print('angle: ', angle)
        center_f = [0.0, 0.0]
        matrix = tf._get_inverse_affine_matrix(center_f, -angle, [0.0, 0.0], 1.0, [0.0, 0.0])
        return F_t.rotate(file, matrix=matrix, resample=self.resample,
                            expand=self.expand, fill=self.fill)
        

def _setup_angle(x):
    if isinstance(x, numbers.Number):
        x = [-x, x]

    return [float(d) for d in x]



if __name__=='__main__':
    a = torch.arange(16).reshape(1, 4, 4)
    # f = Boxcar_smooth(3)
    f = RandomHorizontalFlip(0.5)
    b = f(a)
    print(f'a:\n{a}\nb:\n{b}')
    print('done')