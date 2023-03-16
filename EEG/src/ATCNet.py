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
        self, n_classes = 4, in_chans = 22, in_samples = 1000, n_windows = 5, 
        eegn_F1 = 16, eegn_D = 2, eegn_kernelSize = 64, eegn_poolSize = 7, eegn_dropout=0.3, 
        tcn_depth = 2, tcn_kernelSize = 4, tcn_filters = 32, tcn_dropout = 0.3, 
        tcn_activation = 'elu', fuse = 'average', MSA_embed_dim = 8, MSA_num_heads = 2, MSA_dropout=0.5
    ):
        super().__init__()
        #some hyperparams to remember
        self.input_shape = (in_chans, in_samples)
        self.n_windows = n_windows
        self.fuse = fuse
        self.Filter_Num_2 = eegn_F1*eegn_D

        #EEGNet block
        self.Convolution_module = Convolution_module(
            n_temporal_filters=eegn_F1, 
            kernel_length=eegn_kernelSize, 
            pool_size2=eegn_poolSize, 
            depth_multiplier=eegn_D, 
            in_channels=in_chans, 
            dropout=eegn_dropout
        )
        
        #attention block for each window
        self.attention_modules = [nn.MultiheadAttention(
            embed_dim=self.Filter_Num_2, 
            # kdim=MSA_embed_dim, 
            # vdim=MSA_embed_dim, 
            num_heads=MSA_num_heads, 
            dropout=MSA_dropout,
            batch_first=True) for _ in range(self.n_windows)]
        self.attention_modules = nn.ModuleList(self.attention_modules)

        #TCN block for each window
        self.TCN_modules = [TCN(
            input_dimension=self.Filter_Num_2,
            depth=tcn_depth,
            kernel_size=tcn_kernelSize,
            filters=tcn_filters,
            dropout=tcn_dropout,
            activation=tcn_activation
        ) for _ in range(self.n_windows)]
        self.TCN_modules = nn.ModuleList(self.TCN_modules)

        #FC layer for each window
        #FIXME: did NOT apply norm constrain here
        self.dense = [nn.LazyLinear(n_classes) for _ in range(self.n_windows)]
        self.dense = nn.ModuleList(self.dense)

    def forward(self, x):
        h = x
        h = h.view(-1, 1, self.input_shape[0], self.input_shape[1])
        #EEGNet block
        h = self.Convolution_module(h) #(64, 32, 1, 15)
        h = h[:,:,-1,:] #(64, 32, 15)
        h = h.view(-1, h.shape[2], h.shape[1]) #(64, 15, 32)

        sliding_windows_concat = []
        for i in range(self.n_windows):
            start = i
            end = h.shape[1]-self.n_windows+i+1
            sub_h = h[:, start:end, :]
            #attention
            sub_h, _ = self.attention_modules[i](sub_h, sub_h, sub_h)
            #TCN block
            sub_h = self.TCN_modules[i](sub_h)
            sub_h = sub_h[:,-1,:]
            if (self.fuse == 'average'):
                sub_h = self.dense[i](sub_h)
                sliding_windows_concat.append(sub_h)
            else:
                raise NotImplementedError
        
        if (self.fuse == 'average'):
            out = torch.stack(sliding_windows_concat, dim=1)
            out = torch.mean(out, dim=1)
        else:
            raise NotImplementedError

        return out

class Convolution_module(nn.Module):
    def __init__(
        self, n_temporal_filters=4, 
        kernel_length=64, pool_size1=8, pool_size2=7, 
        depth_multiplier=2, in_channels=22, dropout=0.1):

        '''
        Default hyperparameters s1:
        self, n_temporal_filters=4, 
        kernel_length=64, pool_size=8, 
        depth_multiplier=2, in_channels=22, dropout=0.1
        '''

        '''
        hyperparameters s2:
        self, n_temporal_filters=8, 
        kernel_length=64, pool_size=8, 
        depth_multiplier=4, in_channels=22, dropout=0.3
        '''

        '''
        hyperparameters s3:
        self, n_temporal_filters=12, 
        kernel_length=64, pool_size=8, 
        depth_multiplier=6, in_channels=22, dropout=0.4
        '''

        '''
        hyperparameters s4:
        self, n_temporal_filters=16, 
        kernel_length=64, pool_size=8, 
        depth_multiplier=8, in_channels=22, dropout=0.4
        '''

        super().__init__()

        kernel_length2 = 16
        Filter_Num_2 = depth_multiplier*n_temporal_filters

        self.temporal_conv1 = nn.Conv2d(1, n_temporal_filters, (1,kernel_length), padding='same', bias=False)
        self.batch_norm_1 = nn.BatchNorm2d(n_temporal_filters)
        self.depth_wise_conv = nn.Conv2d(n_temporal_filters, Filter_Num_2, (in_channels, 1), bias=False, groups=n_temporal_filters)
        self.batch_norm_2 = nn.BatchNorm2d(Filter_Num_2)
        self.elu = nn.ELU()
        self.average_pool1 = nn.AvgPool2d((1, pool_size1), stride=(1, pool_size1))
        self.average_pool2 = nn.AvgPool2d((1, pool_size2), stride=(1, pool_size2))
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.spatial_conv1 = nn.Conv2d(Filter_Num_2, Filter_Num_2, (1, kernel_length2), padding='same', bias=False)
        self.batch_norm_3 = nn.BatchNorm2d(Filter_Num_2)

        #NOTE: remove this if used as part of ATCNet, keep this if used as EGGNet
        #self.temp_linear = nn.LazyLinear(4)

    def forward(self, x):
        # x should be (batch_size, 1, 22, 1000)
        h = x
        h = self.temporal_conv1(h)
        h = self.batch_norm_1(h)
        h = self.depth_wise_conv(h)
        h = self.batch_norm_2(h)
        h = self.elu(h)
        h = self.average_pool1(h)
        h = self.dropout1(h)
        h = self.spatial_conv1(h)
        h = self.batch_norm_3(h)
        h = self.elu(h)
        h = self.average_pool2(h)
        h = self.dropout2(h)

        #NOTE: remove this if used as part of ATCNet, keep this if used as EGGNet
        # h=h.view(h.shape[0], -1)
        # h=self.temp_linear(h)

        return h #(64, 32, 1, 15)

class TCN(nn.Module):   
    def __init__(
        self, input_dimension, depth=2, 
        kernel_size=4, filters=32, 
        dropout=0.3, activation='elu'):

        super().__init__()

        dilation = 1
        padding = kernel_size - 1 # casual padding
        self.depth = depth

        #NOTE: For each depth, the dilation is different
        self.conv1_list = [
            nn.Conv1d(
            in_channels = input_dimension,  
            out_channels = filters, 
            kernel_size = kernel_size, 
            padding = (kernel_size - 1) * (2**(i)), 
            dilation = 2**(i))
            for i in range(self.depth)
        ]
        self.conv1_list = nn.ModuleList(self.conv1_list)

        self.conv2_list = [
            nn.Conv1d(
            in_channels = input_dimension,  
            out_channels = filters, 
            kernel_size = kernel_size, 
            padding = (kernel_size - 1) * (2**(i)), 
            dilation = 2**(i))
            for i in range(self.depth)
        ]
        self.conv2_list = nn.ModuleList(self.conv2_list)

        self.pad_list = [(kernel_size-1)*(2**(i)) for i in range(self.depth)]

        self.batch_norm_list1 = [nn.BatchNorm1d(filters) for _ in range(self.depth)]
        self.batch_norm_list1 = nn.ModuleList(self.batch_norm_list1)
        self.batch_norm_list2 = [nn.BatchNorm1d(filters) for _ in range(self.depth)]
        self.batch_norm_list2 = nn.ModuleList(self.batch_norm_list2)

        self.dropout_list1 = [nn.Dropout(p=dropout) for _ in range(self.depth)]
        self.dropout_list1 = nn.ModuleList(self.dropout_list1)
        self.dropout_list2 = [nn.Dropout(p=dropout) for _ in range(self.depth)]
        self.dropout_list1 = nn.ModuleList(self.dropout_list1)

        if activation == 'elu':  
            self.elu = nn.ELU()
        else:
            raise NotImplementedError

        for conv in self.conv1_list:
            torch.nn.init.kaiming_uniform(conv.weight)
        for conv in self.conv2_list:
            torch.nn.init.kaiming_uniform(conv.weight)

        # self.conv1 = nn.Conv1d(
        #     in_channels = input_dimension, 
        #     out_channels = filters, 
        #     kernel_size = kernel_size, 
        #     padding = padding_dilation_2, 
        #     dilation = dilation+1)
        # self.conv1_2 = nn.Conv1d(
        #     in_channels = input_dimension, 
        #     out_channels = filters, 
        #     kernel_size = kernel_size, 
        #     padding = padding_dilation_2, 
        #     dilation = dilation+1)
        # torch.nn.init.kaiming_uniform(self.conv1.weight)
        # self.casual_conv1 = lambda x: x[:, :, :-padding].contiguous()
        # self.casual_conv1_2 = lambda x: x[:, :, :-padding_dilation_2].contiguous()
        # self.batch_norm_1 = nn.BatchNorm1d(filters)
        # if activation == 'elu':  
        #     self.elu = nn.ELU()
        # self.dropout1 = nn.Dropout(p=dropout)
        # self.conv2 = nn.Conv1d(
        #     in_channels = input_dimension,  
        #     out_channels = filters, 
        #     kernel_size = kernel_size, 
        #     padding = padding, 
        #     dilation = dilation)
        # self.conv2_2 = nn.Conv1d(
        #     in_channels = input_dimension, 
        #     out_channels = filters, 
        #     kernel_size = kernel_size, 
        #     padding = padding_dilation_2, 
        #     dilation = dilation+1)
        # torch.nn.init.kaiming_uniform(self.conv2.weight)
        # self.casual_conv2 = lambda x: x[:, :, :-padding].contiguous()
        # self.casual_conv2_2 = lambda x: x[:, :, :-padding_dilation_2].contiguous()
        # self.batch_norm_2 = nn.BatchNorm1d(filters)
        # self.dropout2 = nn.Dropout(p=dropout)

        if input_dimension != filters:
            self.convInput = nn.Conv1D(input_dimension, filters, kernel_size=1, padding='same')
            torch.nn.init.kaiming_uniform(self.convInput.weight)
        else:
            self.convInput = None
    
    def forward(self, x):
        
        h = x #(64, 13, 32)
        x = x.reshape(-1, x.shape[2], x.shape[1]) #(64, 32, 13)
        h = h.reshape(-1, h.shape[2], h.shape[1]) #(64, 32, 13)
        h = self.conv1_list[0](h)
        #h = self.casual_conv_list[0](h)
        h = h[:, :, :-self.pad_list[0]]
        h = self.batch_norm_list1[0](h)
        h = self.elu(h)
        h = self.dropout_list1[0](h)
        h = self.conv2_list[0](h)
        #h = self.casual_conv_list[0](h)
        h = h[:, :, :-self.pad_list[0]]
        h = self.batch_norm_list2[0](h)
        h = self.elu(h)
        h = self.dropout_list2[0](h)
        if self.convInput is None:
            h = h + x
        else:
            h = h + self.convInput(x)
        out = self.elu(h)

        for i in range(1, self.depth-1):
            h = self.conv1_list[i](out)
            #h = self.casual_conv_list[i](h)
            h = h[:, :, :-self.pad_list[i]]
            h = self.batch_norm_list1[i](h)
            h = self.elu(h)
            h = self.dropout_list1[i](h)
            h = self.conv2_list[i](h)
            #h = self.casual_conv_list[i](h)
            h = h[:, :, :-self.pad_list[i]]
            h = self.batch_norm_list2[i](h)
            h = self.elu(h)
            h = self.dropout_list2[i](h)
            h = h + out
            out = self.elu(h)

        out = out.reshape(-1, out.shape[2], out.shape[1]) #(64, 13, 32)

        return out









