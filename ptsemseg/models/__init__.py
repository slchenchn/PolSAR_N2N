'''
Author: Shuailin Chen
Created Date: 2020-11-27
Last Modified: 2021-05-21
	content: 
'''
import copy
import torchvision.models as models

from ptsemseg.models.unetpp import UnetPP

def get_model(model_dict):
    model_dict = model_dict.to_flatten_dict()
    name = model_dict['arch']
    model = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)      
    param_dict.pop('arch')
    
    model = model(**param_dict)

    return model


def _get_model_instance(name):
    try:
        return {
            "unetpp": UnetPP,
        }[name]
    except:
        raise("Model {} not available".format(name))
