'''
Author: Shuailin Chen
Created Date: 2021-05-22
Last Modified: 2021-05-22
	content: 
'''

def N2N_loss(noisy_sub1_denoised, 
            noisy_sub2_denoised):
  ''' Neighbor2Neighbor loss, undone
  '''
  noisy_target = noisy_sub2
  if cfg.train.loss.gamma.const:
      gamma = cfg.train.loss.gamma.base
  else:
      gamma = it / train_iter * cfg.train.loss.gamma.base

  diff = noisy_output - noisy_target
  exp_diff = noisy_sub1_denoised - noisy_sub2_denoised
  loss1 = torch.mean(diff**2)
  loss2 = gamma * torch.mean((diff - exp_diff)**2)
  loss_all = loss1 + loss2
  loss_all.backward()
