'''
Author: Shuailin Chen
Created Date: 2021-05-22
Last Modified: 2021-05-22
	content: 
'''
from torch import Tensor
import numpy as np
from random import shuffle


def rand_pool(img, mask=None):
    ''' Random pooling, this version is erroneous, cause it wasn't stochastic, the distribution of chosen positions isn't uniform. Which can be reproduced by:
    ```
        h = 100
        img = np.arange(h**2).reshape(h, h, 1)
        img = np.tile(img, (1, 1, 3))
        sub_imgs, mask = PolSAR.rand_pool(img)

        unique, counts = np.unique(mask[0], return_counts=True)
        print(f'hist of mask 0: {dict(zip(unique, counts))}')

        unique, counts = np.unique(mask[1], return_counts=True)
        print(f'hist of mask 1: {dict(zip(unique, counts))}')
    ```

    Args:
        img (ndarray): image file, in shape of [height, weight, channel],
            whose shape should be divisible by 2
        mask (list): subsample mask, if None, this func generates one. 
            Default: None
    Returns:
        sub_img (list): two subsampled images
        mask (list): subsample mask
    '''

    # check variable type
    if isinstance(img, Tensor):
        img = img.numpy()

    # generate mask if None
    h, w, c = img.shape
    if mask is None:
        pool_enc = np.random.randint(0, 6, size=(h//2, w//2))
        pool_dec_1 = (pool_enc-1) // 2
        pool_dec_1[pool_dec_1<0] = 0
        pool_dec_2 = pool_enc%3 + pool_dec_1 + 1
        pool_dec_2[pool_enc==5] = 3
        mask = [pool_dec_1.flatten(), pool_dec_2.flatten()]
        shuffle(mask)

    # apply subsample mask
    img_ = img.reshape(h//2, 2, w//2, 2, c).transpose(0, 2, 1, 3, 4).reshape(h//2, w//2, 4, c)
    h_idx = np.arange(h//2).repeat(w//2)
    w_idx = np.tile(np.arange(w//2), h//2)
    sub_imgs = []
    sub_imgs.append(img_[h_idx, w_idx, mask[0], :].reshape(h//2, w//2, c))
    sub_imgs.append(img_[h_idx, w_idx, mask[1], :].reshape(h//2, w//2, c))

    return sub_imgs, mask