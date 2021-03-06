'''
Author: Shuailin Chen
Created Date: 2020-11-27
Last Modified: 2021-07-05
	content: 
'''
import copy
import torchvision.models as models

from ptsemseg.models.unetpp import UnetPP
from ptsemseg.models.unetpp2 import UnetPP2
from ptsemseg.models.unetpp3 import UnetPP3
from ptsemseg.models.unetpp4 import UnetPP4
from ptsemseg.models.unetpp5 import UnetPP5


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
            "unetpp2": UnetPP2,
            "unetpp3": UnetPP3,
            "unetpp4": UnetPP4,
            "unetpp5": UnetPP5,
        }[name]
    except:
        raise("Model {} not available".format(name))
