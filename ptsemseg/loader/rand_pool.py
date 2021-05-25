'''
Author: Shuailin Chen
Created Date: 2021-05-22
Last Modified: 2021-05-25
	content: adapt from https://github.com/TaoHuang2018/Neighbor2Neighbor/blob/main/training_script.md
'''

import torch
import numpy as np

operation_seed_counter = 0


def get_generator(device='cuda'):
    global operation_seed_counter
    operation_seed_counter += 1
    g_cuda_generator = torch.Generator(device=device)
    g_cuda_generator.manual_seed(operation_seed_counter)
    return g_cuda_generator


def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size**2, h // block_size,
                           w // block_size)


def generate_mask_pair(img, device='cuda'):
    # This function generates random masks with shape (N x C x H/2 x W/2)
    n, c, h, w = img.shape
    mask1 = torch.zeros(size=(n * h // 2 * w // 2 * 4, ),
                        dtype=torch.bool,
                        device=img.device)
    mask2 = torch.zeros(size=(n * h // 2 * w // 2 * 4, ),
                        dtype=torch.bool,
                        device=img.device)
    idx_pair = torch.tensor(
        [[0, 1], [0, 2], [1, 3], [2, 3], [1, 0], [2, 0], [3, 1], [3, 2]],
        dtype=torch.int64,
        device=img.device)
    rd_idx = torch.zeros(size=(n * h // 2 * w // 2, ),
                         dtype=torch.int64,
                         device=img.device)
    torch.randint(low=0,
                  high=8,
                  size=(n * h // 2 * w // 2, ),
                  generator=get_generator(device),
                  out=rd_idx)
    rd_pair_idx = idx_pair[rd_idx]
    rd_pair_idx += torch.arange(start=0,
                                end=n * h // 2 * w // 2 * 4,
                                step=4,
                                dtype=torch.int64,
                                device=img.device).reshape(-1, 1)
    mask1[rd_pair_idx[:, 0]] = 1
    mask2[rd_pair_idx[:, 1]] = 1
    return mask1, mask2
   

def generate_subimages(img, mask):
    # This function generates paired subimages given random masks
    n, c, h, w = img.shape
    subimage = torch.zeros(n,
                           c,
                           h // 2,
                           w // 2,
                           dtype=img.dtype,
                           layout=img.layout,
                           device=img.device)
    for i in range(c):
        img_per_channel = space_to_depth(img[:, i:i + 1, :, :], block_size=2)
        img_per_channel = img_per_channel.permute(0, 2, 3, 1).reshape(-1)
        subimage[:, i:i + 1, :, :] = img_per_channel[mask].reshape(
            n, h // 2, w // 2, 1).permute(0, 3, 1, 2)
    return subimage


if __name__ == '__main__':
    # a = np.arange(4).reshape(2,2)
    # r = 100
    # a = np.tile(a[np.newaxis, np.newaxis, :, :], (2, 2, r, r))
    a = np.ones(shape=(4, 3, 320, 480))
    # print(f'before:\n', a)

    # a = a.reshape(1, 2, 2*r, 2*r)
    a = a.astype(np.float32)
    a = torch.from_numpy(a)
    a = a.to('cuda')
    # a = a.to(dtype=torch.int32, device='cuda')
    print(f'device: {a.device}, dtype: {a.dtype}')
    m1, m2 = generate_mask_pair(a, device='cuda')
    sub1 = generate_subimages(a, m1).cpu().numpy()
    sub2 = generate_subimages(a, m2).cpu().numpy()

    
    unique, counts = np.unique(sub1, return_counts=True)
    print(f'hist of sub 0: {dict(zip(unique, counts))}')

    unique, counts = np.unique(sub2, return_counts=True)
    print(f'hist of sub 0: {dict(zip(unique, counts))}')