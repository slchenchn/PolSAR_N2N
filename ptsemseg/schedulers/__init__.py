'''
Author: Shuailin Chen
Created Date: 2021-04-24
Last Modified: 2021-05-31
	content: 
'''
import logging 
from torch.optim import lr_scheduler
from ptsemseg.schedulers.schedulers import *

logger = logging.getLogger('ptsemseg')

key2scheduler = {'constant_lr': ConstantLR,
                 'poly_lr': PolynomialLR,
                 'poly_lr_chen': PolynomialLR_chen,
                 'multi_step': MultiStepLR,
                 'step': lr_scheduler.StepLR,
                 'cosine_annealing': CosineAnnealingLR,
                 'exp_lr': ExponentialLR}


def get_scheduler(optimizer, scheduler_dict):
    scheduler_dict = scheduler_dict.to_flatten_dict()
    if scheduler_dict is None:
        logger.info('Using No LR Scheduling')
        return ConstantLR(optimizer)
    
    s_type = scheduler_dict['name']
    scheduler_dict.pop('name')

    logging.info('Using {} learning rate scheduler with {} params'.format(s_type,
                                                            scheduler_dict))

    warmup_dict = {} 
    if 'warmup_iters' in scheduler_dict:
        # This can be done in a more pythonic way... 
        warmup_dict['warmup_iters'] = scheduler_dict.get('warmup_iters', 100)
        warmup_dict['mode'] = scheduler_dict.get('warmup_mode', 'linear')
        warmup_dict['gamma'] = scheduler_dict.get('warmup_factor', 0.2)

        logger.info('Using Warmup with {} iters {} gamma and {} mode'.format(
                                        warmup_dict['warmup_iters'],
                                        warmup_dict['gamma'],
                                        warmup_dict['mode']))

        scheduler_dict.pop('warmup_iters', None) 
        scheduler_dict.pop('warmup_mode', None)
        scheduler_dict.pop('warmup_factor', None) 

        base_scheduler = key2scheduler[s_type](optimizer, **scheduler_dict)
        return WarmUpLR(optimizer, base_scheduler, **warmup_dict)

    return key2scheduler[s_type](optimizer, **scheduler_dict)
