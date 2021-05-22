'''
Author: Shuailin Chen
Created Date: 2021-04-25
Last Modified: 2021-05-22
	content: 
'''
import cv2
from mylib import my_torch_tools as mt
from ptsemseg.augmentations.augmentations import *
from ptsemseg.loader.polsar import *
import torch
from ptsemseg.loader import rand_pool

_TMP_PATH = './tmp'


if __name__=='__main__':
	aug = Compose([RandomHorizontalFlip(0.5), RandomVerticalFlip(0.5)])
	ds = PolSAR(file_root=r'data/GF3', 
                split_root=r'data/GF3/split/denoise/Hoekman/0.9', 
                split='train', 
                augments=aug,
                data_format='Hoekman',
                norm=True,
                )
	idx = 2342

	img = ds.__getitem__(idx)
	img = torch.unsqueeze(img, 0)
	img = img.to('cuda')
	print(f'device: {img.device}, dtype: {img.dtype}, shape: {img.shape}')

	m1, m2 = rand_pool.generate_mask_pair(img, device='cuda')
	sub1 = rand_pool.generate_subimages(img, m1).cpu().numpy()
	sub2 = rand_pool.generate_subimages(img, m2).cpu().numpy()

	img = img.cpu().numpy().squeeze()
	sub1 = sub1.squeeze()
	sub2 = sub1.squeeze()
	img = mt.Tensor2cv2image(img)[0]
	sub1 = mt.Tensor2cv2image(sub1)[0]
	sub2 = mt.Tensor2cv2image(sub2)[0]
	sub_imgs = [sub1, sub2]

	print(f'\nimg shape: {img.shape}, sub img shape: {sub_imgs[0].shape}, {sub_imgs[1].shape}\n')
	for ii in range(3):
		cv2.imwrite(osp.join(_TMP_PATH, f'{ii}.png'), img[..., ii*3: ii*3+3])

	for jj, sub_img in enumerate(sub_imgs):
		for ii in range(3):
			cv2.imwrite(osp.join(_TMP_PATH, f'sub_{jj}_{ii}.png'), sub_img[..., ii*3: ii*3+3])





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
