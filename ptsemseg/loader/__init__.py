'''
Author: Shuailin Chen
Created Date: 2021-04-24
Last Modified: 2021-04-25
	content: 
'''

from ptsemseg.loader.polsar import *

def get_loader(name):
    """get_loader

    :param name:
    """
    return {
        'PolSAR', PolSAR,
    }[name]