'''
Author: Shuailin Chen
Created Date: 2021-04-25
Last Modified: 2021-07-03
	content: 
'''
import cv2
from mylib import my_torch_tools as mt
from ptsemseg.augmentations.augmentations import *
from ptsemseg.loader.polsar_simulate import *
import torch
from ptsemseg.loader import rand_pool
from mylib import simulate

_TMP_PATH = './tmp'



if __name__=='__main__':

	''' test generate_Wishart_noise_from_img() '''
	R = 15
	a = R * np.ones((10, 10, 3))
	ENL = 300
	img, noise = simulate.generate_Wishart_noise_from_img(a, ENL)

	# noise = psr.Hokeman_decomposition(noise)
	noise = np.real(noise[0, ...])
	noise = np.log(noise)

	print(np.log(R), noise.mean(), R**2/ENL, noise.var())



	''' test inverse_Hoekman_decomposition '''
	# path = r'/home/csl/code/PolSAR_N2N/data/SAR_CD/GF3/data/E115_N39_中国河北/降轨/1/20161209/C3'
	# C3 = psr.read_c3(path, out='save_space')
	# hoek = psr.Hokeman_decomposition(C3)
	# C3_new = psr.inverse_Hokeman_decomposition(hoek)
	# err = ((C3-C3_new)**2)
	# print(err.min())
	# print(err.max())
	# print(err.mean())
	# print(C3.dtype, C3.shape)


	''' test dataloader '''
	# aug = Compose([RandomHorizontalFlip(0.5), RandomVerticalFlip(0.5)])
	# ds = PolSARSimulate(file_root=r'data/BSR/BSDS500/data/images', 
    #             split_root=r'data/GF3/split/denoise/Hoekman/0.9', 
    #             split='train', 
    #             augments=aug,
    #             data_format='Hoekman',
    #             norm=False,
	# 			ENL=1,
	# 			log = False,
    #             )
	# idx = 309

	# img, noise, file_path = ds.__getitem__(idx)

	# zz = torch.stack((img, noise), dim=0)
	# zz = ds.Hoekman_recover_to_C3(zz)
	# img = zz[0, ...]
	# noise = zz[1, ...]
	# # img = ds.Hoekman_recover_to_C3(img)
	# # noise = ds.Hoekman_recover_to_C3(noise)

	# pauli_img = psr.rgb_by_c3(img, type='sinclair')
	# pauli_noise = psr.rgb_by_c3(noise, type='sinclair')

	# cv2.imwrite(osp.join(_TMP_PATH, 'rpauli_img.jpg'), cv2.cvtColor((255*pauli_img).astype(np.uint8), cv2.COLOR_RGB2BGR))
	# cv2.imwrite(osp.join(_TMP_PATH, 'rpauli_noise.jpg'), cv2.cvtColor((255*pauli_noise).astype(np.uint8), cv2.COLOR_RGB2BGR))
	# print(f'file path: {file_path}')


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
