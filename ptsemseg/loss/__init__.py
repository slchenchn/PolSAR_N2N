import copy
import logging
import functools
import torch.nn.functional as F

from ptsemseg.loss.loss import *
# from ptsemseg.loss.loss import cross_entropy2d
# from ptsemseg.loss.loss import bootstrapped_cross_entropy2d
# from ptsemseg.loss.loss import multi_scale_cross_entropy2d
# from ptsemseg.loss.loss import FocalLoss
# from ptsemseg.loss.loss import FocalLoss2d


logger = logging.getLogger('ptsemseg')

key2loss = {'cross_entropy2d': cross_entropy2d,
            'cross_entropy': cross_entropy1d,
            'bootstrapped_cross_entropy': bootstrapped_cross_entropy2d,
            'multi_scale_cross_entropy': multi_scale_cross_entropy2d,
            'focal_loss':FocalLoss,
            'focal_loss_2d':FocalLoss2d,
            'cross_entropy_mask':cross_entropy_with_mask,
            }

def get_loss_function(cfg):
    if cfg.train.loss is None:
        logger.info("Using default cross entropy loss")
        return cross_entropy2d

    else:
        loss_dict = cfg.train.loss
        loss_name = loss_dict.name
        loss_params = {k:v for k,v in vars(loss_dict).items() if k != 'name'}

        if loss_name not in key2loss:
            raise NotImplementedError('Loss {} not implemented'.format(loss_name))

        logger.info('Using {} loss with {} params'.format(loss_name, 
                                                     loss_params))
        return functools.partial(key2loss[loss_name], **loss_params)
