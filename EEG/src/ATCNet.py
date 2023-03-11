import logging
import argparse
from re import X
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl 
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchmetrics.functional import accuracy

class ATCNet(nn.Module):
    """ 
    ATCNet model from Altaheri et al 2022.
    See details at https://ieeexplore.ieee.org/abstract/document/9852687

    Notes
    -----
    The initial values in this model are based on the values identified by
    the authors
    
    References
    ----------
    .. H. Altaheri, G. Muhammad and M. Alsulaiman, "Physics-informed 
        attention temporal convolutional network for EEG-based motor imagery 
        classification," in IEEE Transactions on Industrial Informatics, 2022, 
        doi: 10.1109/TII.2022.3197419.
    """
    def __init__(
        self, n_classes = 4, in_chans = 22, in_samples = 1000, n_windows = 3, attention = None, 
        eegn_F1 = 16, eegn_D = 2, eegn_kernelSize = 64, eegn_poolSize = 8, eegn_dropout=0.3, 
        tcn_depth = 2, tcn_kernelSize = 4, tcn_filters = 32, tcn_dropout = 0.3, 
        tcn_activation = 'elu', fuse = 'average'
    ):
        super().__init__()
        self.Convolution_module = Convolution_module()
        self.input_shape = (in_chans, in_samples)

    def forward(self, x):
        h = x
        h = h.view(-1, 1, self.input_shape[0], self.input_shape[1])
        h = self.Convolution_module(h)

        return h

class Convolution_module(nn.Module):
    def __init__(
        self, n_temporal_filters=4, 
        kernel_length=64, pool_size=8, 
        depth_multiplier=2, in_channels=22, dropout=0.1):

        super().__init__()

        kernel_length2 = 16
        Filter_Num_2 = depth_multiplier*n_temporal_filters

        self.temporal_conv1 = nn.Conv2d(1, n_temporal_filters, (1,kernel_length), padding='same', bias=False)
        self.batch_norm_1 = nn.BatchNorm2d(n_temporal_filters)
        self.depth_wise_conv = nn.Conv2d(n_temporal_filters, Filter_Num_2, (in_channels, 1), bias=False, groups=n_temporal_filters)
        self.batch_norm_2 = nn.BatchNorm2d(Filter_Num_2)
        self.elu = nn.ELU()
        self.average_pool = nn.AvgPool2d((1, pool_size), stride=(1, pool_size))
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.spatial_conv1 = nn.Conv2d(Filter_Num_2, Filter_Num_2, (1, kernel_length2), padding='same', bias=False)
        self.batch_norm_3 = nn.BatchNorm2d(Filter_Num_2)

        #TODO: remove this
        self.temp_linear = nn.LazyLinear(4)

    def forward(self, x):
        # x should be (batch_size, 1, 22, 1000)
        h = x
        h = self.temporal_conv1(h)
        h = self.batch_norm_1(h)
        h = self.depth_wise_conv(h)
        h = self.batch_norm_2(h)
        h = self.elu(h)
        h = self.average_pool(h)
        h = self.dropout1(h)
        h = self.spatial_conv1(h)
        h = self.batch_norm_3(h)
        h = self.elu(h)
        h = self.average_pool(h)
        h = self.dropout2(h)

        #TODO: remove this
        h=h.view(h.shape[0], -1)
        h=self.temp_linear(h)

        return h

# class TCN(nn.Module):
#      def __init__(
#          self, 
#          input_dimension,
#          depth=2,
#          kernel_size=4,
#          filters=32,
#          dropout=0.3,
#          activation='elu'):





