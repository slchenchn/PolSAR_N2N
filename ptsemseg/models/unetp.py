'''
Author: Shuailin Chen
Created Date: 2021-05-21
Last Modified: 2021-05-21
	content: inherit from unet.py, add 3 1x1 conv layer to the output
    note: undone
'''
import torch.nn as nn

from ptsemseg.models.utils import *
from ptsemseg.models.unet import unet

class unetp(unet):
    def __init__(
        self,
        feature_scale=4,
        n_classes=21,
        is_deconv=True,
        in_channels=3,
        is_batchnorm=True,
    ):
        super().__init__(feature_scale, n_classes, is_deconv, in_channels, is_batchnorm)

        self.final2 = nn.Conv2d(n_classes, n_classes, 1)
    


    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)

        return final
