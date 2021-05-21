'''
Author: Shuailin Chen
Created Date: 2020-11-27
Last Modified: 2021-04-25
	content: 
'''
import logging
from ptsemseg.augmentations.augmentations import *
from mylib.nestargs import NestedNamespace

logger = logging.getLogger('ptsemseg')

key2aug = {
           'hflip': RandomHorizontalFlip,
           'vflip': RandomVerticalFlip,
           'rotate': RandomRotation,
        #    'boxcar': BoxcarSmooth,
        #    'gamma': AdjustGamma,
        #    'hue': AdjustHue,
        #    'brightness': AdjustBrightness,
        #    'saturation': AdjustSaturation,
        #    'contrast': AdjustContrast,
        #    'rcrop': RandomCrop,
        #    'scale': Scale,
        #    'rsize': RandomSized,
        #    'rsizecrop': RandomSizedCrop,
        #    'translate': RandomTranslate,
        #    'ccrop': CenterCrop,
           }

def get_composed_augmentations(aug_dict):
    if aug_dict is None:
        logger.info("Using No Augmentations")
        return None

    augmentations = []
    for aug_key, aug_param in aug_dict.items():
        if isinstance(aug_param, NestedNamespace):
            aug_param = vars(aug_param)
        augmentations.append(key2aug[aug_key](aug_param))
        logger.info("Using {} aug with params {}".format(aug_key, aug_param))
    return Compose(augmentations)


