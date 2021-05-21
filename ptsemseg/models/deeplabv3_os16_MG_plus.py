# camera-ready

import torch
import torch.nn as nn
import torch.nn.functional as F

import os

from ptsemseg.models.resnet_plus import ResNet18_OS16, ResNet34_OS16, ResNet50_OS16, ResNet101_OS16, ResNet152_OS16, \
    ResNet18_OS8, ResNet34_OS8
# from ptsemseg.models.aspp import ASPP, ASPP_Bottleneck
from ptsemseg.models.aspp_plus import ASPP, ASPP_Bottleneck


class DeepLabV3_MG_plus(nn.Module):
    # def __init__(self, num_classes,model_id, project_dir):
    def __init__(self, num_classes):
        super(DeepLabV3_MG_plus, self).__init__()

        # self.num_classes = 20
        self.num_classes = num_classes

        # self.model_id = model_id
        # self.project_dir = project_dir
        # self.create_model_dirs()

        # self.resnet = ResNet18_OS8() # NOTE! specify the type of ResNet here
        self.resnet = ResNet50_OS16()
        print('backbone: pretrained ResNet50')
        
        self.aspp = ASPP_Bottleneck(
            num_classes=self.num_classes)  # NOTE! if you use ResNet50-152, set self.aspp = ASPP_Bottleneck(num_classes=self.num_classes) instead

        low_channel=48
        normal_channel=256
        self.decoder=decoder(low_channel,normal_channel,self.num_classes)


    def forward(self, x):
        # (x has shape (batch_size, 3, h, w))

        h_upsample_first = int(x.size()[2] / 4)
        w_upsample_first = int(x.size()[3] / 4)
        h_upsample_second = int(x.size()[2])
        w_upsample_second = int(x.size()[3])

        feature_map,low_level_feature = self.resnet(x)  # (shape: (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet is ResNet18_OS8 or ResNet34_OS8, it will be (batch_size, 512, h/8, w/8). If self.resnet is ResNet50-152, it will be (batch_size, 4*512, h/16, w/16))

        high_level_feature = self.aspp(feature_map)  # (shape: (batch_size, num_classes, h/16, w/16))

        output_encoder = F.upsample(high_level_feature, size=(h_upsample_first, w_upsample_first), mode="bilinear")  # (shape: (batch_size, num_classes, h, w))

        output=self.decoder(low_level_feature,output_encoder,h_upsample_second,w_upsample_second)

        return output

class decoder(nn.Module):
    def __init__(self,low_channel,normal_channel,num_classes):
        super(decoder,self).__init__()
        self.low_channel=low_channel
        self.normal_channel=normal_channel
        self.num_classes=num_classes

        self.bn_low = nn.BatchNorm2d(low_channel)
        self.bn_normal = nn.BatchNorm2d(normal_channel)
        # self.bn_concat=nn.BatchNorm2d(low_channel+normal_channel)
        self.relu = nn.ReLU(inplace=True)

        self.c1_1 = nn.Conv2d(256, low_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.c3_1 = nn.Conv2d(low_channel+normal_channel, normal_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.c3_2 = nn.Conv2d(normal_channel, normal_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.c_cls = nn.Conv2d(normal_channel, num_classes, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x_low,x_high,height,width):
        x_low=self.c1_1(x_low)
        x_low=self.bn_low(x_low)
        x_low=self.relu(x_low)
        x_cat=torch.cat([x_low,x_high],1)

        x_cat=self.c3_1(x_cat)
        x_cat=self.bn_normal(x_cat)
        x_cat=self.relu(x_cat)

        x_cat = self.c3_2(x_cat)
        x_cat = self.bn_normal(x_cat)
        x_cat = self.relu(x_cat)

        x_cat=self.c_cls(x_cat)
        output = F.upsample(x_cat, size=(height, width), mode="bilinear")  # (shape: (batch_size, num_classes, h, w))
        return output