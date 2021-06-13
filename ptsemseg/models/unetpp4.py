'''
Author: Shuailin Chen
Created Date: 2021-05-21
Last Modified: 2021-06-03
	content: PyTorch implementation of U-Net model for N2N and SSDN.
    adapted from https://github.com/COMP6248-Reproducability-Challenge/selfsupervised-denoising/blob/master-with-report/ssdn/ssdn/models/noise_network.py
    note: this network is partially asymmetric
'''


import torch
import torch.nn as nn

from torch import Tensor


class UnetPP4(nn.Module):
    """Custom U-Net architecture for Self Supervised Denoising (SSDN) and Noise2Noise (N2N).
    Base N2N implementation was made with reference to @joeylitalien's N2N implementation.
    Changes made are removal of weight sharing when blocks are reused. Usage of LeakyReLu over standard ReLu and incorporation of blindspot  unctionality.
    Unlike other typical U-Net implementations dropout is not used when the model is trained.
    Args:
        in_channels (int, optional): Number of input channels, this will 
            typically be either 1 (Mono) or 3 (RGB) but can be more. Defaults to 3.
        zero_output_weights (bool, optional): Whether to initialise the 
            weights of `nin_c` to zero. This is not mentioned in literature but is done as part of the tensorflow implementation for the parameter estimation network. Defaults to False.
    """

    def __init__(
        self,
        in_channels: int = 3,
        zero_output_weights: bool = False,
        if_print=True
    ):
        if if_print:
            print(f'model UnetPP:\n\tin_channels={in_channels}\n\tzero_output_weights={zero_output_weights}')
            
        super().__init__()
        self._zero_output_weights = zero_output_weights
        self.Conv2d = nn.Conv2d

        ####################################
        # Encode Blocks
        ####################################

        def _max_pool_block(max_pool: nn.Module) -> nn.Module:
            return max_pool

        # Layers: enc_conv0, enc_conv1, pool1
        self.encode_block_1 = nn.Sequential(
            self.Conv2d(in_channels, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            self.Conv2d(48, 48, 3, padding=1),
            nn.ReLU(inplace=True),
            _max_pool_block(nn.MaxPool2d(2)),
        )

        # Layers: enc_conv(i), pool(i); i=2..5
        def _encode_block_2_3_4_5() -> nn.Module:
            return nn.Sequential(
                self.Conv2d(48, 48, 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                _max_pool_block(nn.MaxPool2d(2)),
            )

        # Separate instances of same encode module definition created
        self.encode_block_2 = _encode_block_2_3_4_5()
        self.encode_block_3 = _encode_block_2_3_4_5()
        self.encode_block_4 = _encode_block_2_3_4_5()
        self.encode_block_5 = _encode_block_2_3_4_5()

        # Layers: enc_conv6
        self.encode_block_6 = nn.Sequential(
            self.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        ####################################
        # Decode Blocks
        ####################################
        # Layers: upsample5
        self.decode_block_6 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"))

        # Layers: dec_conv5a, dec_conv5b, upsample4
        self.decode_block_5 = nn.Sequential(
            self.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            self.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )

        # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
        def _decode_block_4_3_2() -> nn.Module:
            return nn.Sequential(
                self.Conv2d(144, 96, 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                self.Conv2d(96, 96, 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode="nearest"),
            )

        # Separate instances of same decode module definition created
        self.decode_block_4 = _decode_block_4_3_2()
        self.decode_block_3 = _decode_block_4_3_2()
        self.decode_block_2 = _decode_block_4_3_2()

        # Layers: dec_conv1a, dec_conv1b, dec_conv1c,
        self.decode_block_1 = nn.Sequential(
            self.Conv2d(96 + in_channels, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            self.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        ####################################
        # Output Block
        ####################################

        nin_a_io = 96

        # nin_a,b,c, linear_act
        self.output_conv = self.Conv2d(96, in_channels, 1)
        self.output_block = nn.Sequential(
            self.Conv2d(nin_a_io, nin_a_io, 1),
            nn.ReLU(inplace=True),
            self.Conv2d(nin_a_io, 96, 1),
            nn.ReLU(inplace=True),
            self.output_conv,
        )

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """Initializes weights using Kaiming  He et al. (2015).
        Only convolution layers have learnable weights. All convolutions use a leaky relu activation function (negative_slope = 0.1) except the last which is just
        a linear output.
        """
        with torch.no_grad():
            self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, a=0.1)
                m.bias.data.zero_()
        # Initialise last output layer
        if self._zero_output_weights:
            self.output_conv.weight.zero_()
        else:
            nn.init.kaiming_normal_(self.output_conv.weight.data, nonlinearity="linear")

    def forward(self, x: Tensor) -> Tensor:

        # Encoder
        pool1 = self.encode_block_1(x)
        pool2 = self.encode_block_2(pool1)
        pool3 = self.encode_block_3(pool2)
        pool4 = self.encode_block_4(pool3)
        pool5 = self.encode_block_5(pool4)
        encoded = self.encode_block_6(pool5)

        # Decoder
        upsample5 = self.decode_block_6(encoded)
        concat5 = torch.cat((upsample5, pool4), dim=1)
        upsample4 = self.decode_block_5(concat5)
        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self.decode_block_4(concat4)
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self.decode_block_3(concat3)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self.decode_block_2(concat2)
        concat1 = torch.cat((upsample1, x), dim=1)
        x = self.decode_block_1(concat1)

        # Output
        x = self.output_block(x)

        return x