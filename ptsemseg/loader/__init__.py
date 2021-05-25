'''
Author: Shuailin Chen
Created Date: 2021-04-24
Last Modified: 2021-05-25
	content: 
'''

from ptsemseg.loader.polsar import *
from ptsemseg.loader.polsar_simulate import *

def get_loader(name):
    """get_loader

    :param name:
    """
    return {
        'PolSAR': PolSAR,
        'PolSAR_simulate': PolSARSimulate,
    }[name]