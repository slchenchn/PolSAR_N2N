'''
Author: Shuailin Chen
Created Date: 2020-11-27
Last Modified: 2021-04-03
	content: 
'''
import copy
import torchvision.models as models

from ptsemseg.models.fcn import *
from ptsemseg.models.segnet import *
from ptsemseg.models.unet import *
from ptsemseg.models.pspnet import *
from ptsemseg.models.icnet import *
from ptsemseg.models.linknet import *
from ptsemseg.models.frrn import *
from ptsemseg.models.Refine import rf50
from ptsemseg.models.Refine import rf101

# from ptsemseg.models.deeplabv3 import Res_Deeplab
# from ptsemseg.models.MV1_3 import MV1_3_ResNet50
from ptsemseg.models.deeplabv3_os16_MG import DeepLabV3_MG
from ptsemseg.models.deeplabv3_os16_MG_plus import DeepLabV3_MG_plus
from ptsemseg.models.deeplabv3_os16_MG_plus_4band import DeepLabV3_MG_plus_4band
from ptsemseg.models.EffUnet import EffUnet

from ptsemseg.models.MV3_1_true_2 import MV3_1_true_2_ResNet50
# from ptsemseg.models.MV2_9_1 import MV2_10_ResNet50
from ptsemseg.models.MV3_1_true_2_res101 import MV3_1_true_2_ResNet101
from ptsemseg.models.MVD3_1_true_2_os16 import MVD3_1_true_2_os16_ResNet50
from ptsemseg.models.MV3_1_true_2_dropout import MV3_1_true_2_dropout_ResNet50

from siamese.siamunet_diff import SiamUnet_diff
from siamese.siamunet_conc import SiamUnet_conc
from siamese.fresunet import *
from siamese.complex_siamunet_diff import *
from siamese.siamunet_diffv2 import *
from siamese.complex_siamunet_diffv2 import *
from siamese.complex_siamunet_diffv4 import *
from siamese.siam_deeplabv3_diff import *
from siamese.complex_siam_deeplabv3_diff import *
from siamese.surreal_siam_deeplabv3_diff import *
from siamese.surreal_siamunet_diff import *
from siamese.surreal_siamunet_diff2 import *
from siamese.surreal_siamunet_diff3 import *
from siamese.surreal_siamunet_diff4 import *
from siamese.surreal_siamunet_diff5 import *
from siamese.surreal_siamunet_diff6 import *
from ptsemseg.models.my_resnet import *

def get_model(model_dict, n_classes=2):
    model_dict = model_dict.to_flatten_dict()
    name = model_dict['arch']
    model = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)      
    param_dict.pop('arch')

    if name in ["frrnA", "frrnB"]:
        model = model(n_classes, **param_dict)

    elif name in ["fcn32s", "fcn16s", "fcn8s"]:
        model = model(n_classes=n_classes, **param_dict)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)

    elif name == "segnet":
        model = model(n_classes=n_classes, **param_dict)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)

    elif name == "unet":
        model = model(n_classes=n_classes, **param_dict)

    elif name == "pspnet":
        model = model(n_classes=n_classes, **param_dict)

    elif name == "icnet":
        model = model(n_classes=n_classes, **param_dict)

    elif name == "icnetBN":
        model = model(n_classes=n_classes, **param_dict)
    elif name=="refinenet50":
        model=model(num_classes=n_classes,imagenet=True,pretrained=False,**param_dict)
    elif name=="refinenet101":
        model=model(num_classes=n_classes,imagenet=True,pretrained=False,**param_dict)

    elif name in ("deeplabv3_os16_MG", 'siam_deeplabv3_diff', 'complex_siam_deeplabv3_diff', 'surreal_siam_deeplabv3_diff'):
        model=model(num_classes=n_classes, **param_dict)
    elif name=="deeplabv3_os16_MG_plus":
        model=model(num_classes=n_classes, **param_dict)
    elif name=="deeplabv3_os16_MG_plus_4band":
        model=model(num_classes=n_classes, **param_dict)
    # elif name=="deeplabv3":
    #     model=model(num_classes=n_classes)
    elif name=="mv3_res50":
        model=model(num_classes=n_classes, pretrained=False, **param_dict)
    elif name=="mv3_dropout_res50":
        model=model(num_classes=n_classes,**param_dict)
    elif name=="mv3_res101":
        model=model(num_classes=n_classes,**param_dict)
    elif name=="mv2_res50":
        model=model(num_classes=n_classes,**param_dict)
    elif name=="mv1_res50":
        model=model(num_classes=n_classes,**param_dict)
    elif name =="EffUnet":
        model=model()
    elif name in ('siam-diff', 'fresunet', 'fresunet_HM', 'csiam-diff', 'siam-diffv2', 'csiam-diffv2', 'csiam-diffv4', 'surreal_siam_diff', 'surreal_siam_diff2', 'surreal_siam_diff3', 'surreal_siam_diff4', 'surreal_siam_diff5', 'surreal_siam_diff6'):
        model=model(label_nbr=n_classes, **param_dict)
    elif name == 'siam-conc':
        model=model(**param_dict)
    elif name == 'clipped_resnet18':
        model = model(**param_dict)
    else:
        model = model(n_classes=n_classes, **param_dict)

    return model


def _get_model_instance(name):
    try:
        return {
            "fcn32s": fcn32s,
            "fcn8s": fcn8s,
            "fcn16s": fcn16s,
            "unet": unet,
            "segnet": segnet,
            "pspnet": pspnet,
            "icnet": icnet,
            "icnetBN": icnet,
            "linknet": linknet,
            "frrnA": frrn,
            "frrnB": frrn,
            "refinenet50": rf50,
            "refinenet101": rf101,
            "deeplabv3_os16_MG": DeepLabV3_MG,
            "deeplabv3_os16_MG_plus": DeepLabV3_MG_plus,
            "deeplabv3_os16_MG_plus_4band": DeepLabV3_MG_plus_4band,
            # "deeplabv3":Res_Deeplab,
            "mv3_res50": MV3_1_true_2_ResNet50,
            # "mv2_res50": MV2_10_ResNet50,
            "mv3_res101":MV3_1_true_2_ResNet101,
            "mv3_dropout_res50":MV3_1_true_2_dropout_ResNet50,
            "EffUnet":EffUnet,
            'siam-diff':SiamUnet_diff,
            'siam-conc':SiamUnet_conc,
            'clipped_resnet18':clipped_resnet18,
            'fresunet':FresUNet,
            'fresunet_HM':FresUNet_HM,
            'resnet18':models.resnet18,
            'csiam-diff': complex_SiamUnet_diff,
            'siam-diffv2': SiamUnet_diffv2,
            'csiam-diffv2': complex_SiamUnet_diffv2,
            'csiam-diffv4': complex_SiamUnet_diffv4,
            'surreal_siam_diff': surReal_SiamUnet_diff,
            'surreal_siam_diff2': surReal_SiamUnet_diff2,
            'surreal_siam_diff3': surReal_SiamUnet_diff3,
            'surreal_siam_diff4': surReal_SiamUnet_diff4,
            'surreal_siam_diff5': surReal_SiamUnet_diff5,
            'surreal_siam_diff6': surReal_SiamUnet_diff6,
            'siam_deeplabv3_diff': siam_deeplabv3_diff,
            'complex_siam_deeplabv3_diff': complex_siam_deeplabv3_diff,
            'surreal_siam_deeplabv3_diff': surreal_siam_deeplabv3_diff,
        }[name]
    except:
        raise("Model {} not available".format(name))
