'''
Author: Shuailin Chen
Created Date: 2021-04-25
Last Modified: 2021-05-25
	content: 
'''
import cv2
from mylib import my_torch_tools as mt
from ptsemseg.augmentations.augmentations import *
from ptsemseg.loader.polsar_simulate import *
import torch
from ptsemseg.loader import rand_pool

_TMP_PATH = './tmp'


if __name__=='__main__':
	aug = Compose([RandomHorizontalFlip(0.5), RandomVerticalFlip(0.5)])
	ds = PolSARSimulate(file_root=r'data/BSR/BSDS500/data/images', 
                split_root=r'data/GF3/split/denoise/Hoekman/0.9', 
                split='train', 
                augments=aug,
                data_format='Hoekman',
                norm=False,
                )
	idx = 0

	img, noise = ds.__getitem__(idx)

	pauli_img = psr.rgb_by_c3(img, type='sinclair')
	pauli_noise = psr.rgb_by_c3(noise, type='sinclair')

	cv2.imwrite(osp.join(_TMP_PATH, 'pauli_img.jpg'), cv2.cvtColor((255*pauli_img).astype(np.uint8), cv2.COLOR_RGB2BGR))
	cv2.imwrite(osp.join(_TMP_PATH, 'pauli_noise.jpg'), cv2.cvtColor((255*pauli_noise).astype(np.uint8), cv2.COLOR_RGB2BGR))



''' from https://github.com/TaoHuang2018/Neighbor2Neighbor/blob/main/training_script.md '''
# for iteration, clean in enumerate(Dataloader):
#   # preparing synthetic noisy images
#   clean = clean / 255.0
#   clean = clean.cuda()
#   noisy = noise_adder.add_train_noise(clean)
#   optimizer.zero_grad()

#   # generating a sub-image pair
#   mask1, mask2 = generate_mask_pair(noisy)
#   noisy_sub1 = generate_subimages(noisy, mask1)
#   noisy_sub2 = generate_subimages(noisy, mask2)

#   # preparing for the regularization term
#   with torch.no_grad():
#     noisy_denoised = network(noisy)
#   noisy_sub1_denoised = generate_subimages(noisy_denoised, mask1)
#   noisy_sub2_denoised = generate_subimages(noisy_denoised, mask2)

#   # calculating the loss 
#   noisy_output = network(noisy_sub1)
#   noisy_target = noisy_sub2
#   Lambda = epoch / n_epoch * ratio
#   diff = noisy_output - noisy_target
#   exp_diff = noisy_sub1_denoised - noisy_sub2_denoised
#   loss1 = torch.mean(diff**2)
#   loss2 = Lambda * torch.mean((diff - exp_diff)**2)
#   loss_all = loss1 + loss2
#   loss_all.backward()
#   optimizer.step()
