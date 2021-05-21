import copy
import logging
import functools

from torch.optim import SGD
from torch.optim import Adam
from torch.optim import ASGD
from torch.optim import Adamax
from torch.optim import Adadelta
from torch.optim import Adagrad
from torch.optim import RMSprop
from torchlars import LARS

logger = logging.getLogger('ptsemseg')

key2opt =  {'sgd': SGD,
            'adam': Adam,
            'asgd': ASGD,
            'adamax': Adamax,
            'adadelta': Adadelta,
            'adagrad': Adagrad,
            'rmsprop': RMSprop,
            'lars': LARS,}

def get_optimizer(cfg):
    if cfg.train.optimizer is None:
        logger.info("Using SGD optimizer")
        return SGD

    else:
        opt_name = cfg.train.optimizer.name
        if opt_name not in key2opt:
            raise NotImplementedError('Optimizer {} not implemented'.format(opt_name))

        # logger.info('Using {} optimizer'.format(opt_name))
        return key2opt[opt_name]
