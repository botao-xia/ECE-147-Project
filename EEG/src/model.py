import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl 
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchmetrics.classification import MulticlassAccuracy

from utils import PositionalEncoding
from ATCNet import ATCNet, Convolution_module

class ViTransformer(nn.Module):
    def __init__(self, input_shape=(22, 1000), nhead=8, num_encoder_layers=2, patch_width=1, patch_height=22, n_classes=4):
        super().__init__()
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.input_shape = input_shape
        self.flattened_dim = 64

        self.cls_token = nn.Parameter(torch.randn(1,1, self.flattened_dim))
        self.positions = nn.Parameter(torch.randn((input_shape[1] // patch_width) + 1, self.flattened_dim))
        self.flattened_projection = nn.Linear(self.patch_width * self.patch_height, self.flattened_dim)
        self.pos_encoder = PositionalEncoding(self.flattened_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.flattened_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.MLP_head = nn.Linear(self.flattened_dim, n_classes)

        return

    def forward(self, x):
        h = x  #x: (batch_size x 22 x 1000)
        h = h.unsqueeze(1) #(batch_size x 1 x 22 x 1000)
        h = rearrange(h, 'b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=self.patch_height, s2=self.patch_width)  # (batch_size x seq_len x patch_height*patch_width)
        h = self.flattened_projection(h) # (batch_size x seq_len x flattened_dim)
        #add cls token
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=h.shape[0])
        h = torch.cat([cls_tokens, h], dim=1) #(batch_size x seq_len+1 x flattened_dim)
        #add positional encoding
        h += self.positions #(batch_size x seq_len+1 x flattened_dim)
        h = h.view(h.shape[1], h.shape[0], self.flattened_dim) #(seq_len+1 x batch_size x flattened_dim)
        h = self.transformer_encoder(h) # (seq_len+1 x batch_size x flattened_dim)
        h = h.view(h.shape[1], h.shape[0], self.flattened_dim) # (batch_size x seq_len+1 x flattened_dim)
        #h = reduce(h, 'b n e -> b e', reduction='mean')
        h = h[:,0,:]
        h = self.MLP_head(h)
        return h

class ShallowConvNet(nn.Module):
    def __init__(self, input_shape=(22, 1000), n_temporal_filters=40, n_spatial_filters=40, n_classes=4):
        super().__init__() # call __init__ method of superclass
        self.input_shape = input_shape # last two dimensions, (excluding batch size). Should be length 2.
        self.n_temporal_filters = n_temporal_filters
        self.n_spatial_filters = n_spatial_filters
        self.n_classes = n_classes
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        self.temporal_convolution = nn.Conv2d(1, n_temporal_filters, (1, 25))
        # We could implement the spatial convolution as a 1d, or 2d, or 3d convolution.
        # We use 2d here.
        self.spatial_convolution = nn.Conv2d(n_temporal_filters, n_spatial_filters, (input_shape[0], 1))
        self.average_pool = nn.AvgPool2d((1, 75), stride=(1, 15))

        # Final linear layer
        self.n_dense_features = n_spatial_filters*(1 + ((input_shape[1] - 25 + 1) - 75) // 15)
        self.dense = nn.Linear(self.n_dense_features, n_classes)
        # you can also use 'self.dense = nn.LazyLinear(n_classes)' to avoid having to manually compute features
        self.elu = nn.ELU()
        return
    # declaring a forward method also makes the instance a callable.
    # e.g.:
    # model = ShallowConvNet()
    # out = model(x)
    def forward(self, x):
        # x has shape (batch_size, input_shape[0], input_shape[1])
        # Let H0 = input_shape[0], H1 = input_shape[1]
        h = x
        # note that h.view(-1, 1, h.shape[1], h.shape[2]) works normally but does not work with torchinfo
        # this is because the torchinfo input has a weird shape
        h = h.view(-1, 1, self.input_shape[0], self.input_shape[1]) # view as (batch_size, 1, input_shape[0], input_shape[1])
        # Sometimes, view doesn't work and you have to use reshape. This is because of how tensors are stored in memory.
        # 2d convolution takes inputs of shape (batch_size, num_channels, H, W)
        h = self.temporal_convolution(h) # (batch_size, 1, H0, W0) -> (batch_size, n_temporal_filters, H0, W0 - 25 + 1)
        h = self.elu(h)
        h = self.spatial_convolution(h) # (batch_size, n_temporal_filters, H0, W0 - 25 + 1) -> (batch_size, n_spatial_filters, 1, W0 - 25 + 1)
        h = self.elu(h)
        h = h**2 # square
        # alternatively, use torch.pow(h, 2.0)
        h = self.average_pool(h) # (batch_size, n_spatial_filters, 1, W0 - 25 + 1) -> (batch_size, n_spatial_filters, 1, 1 + ((W0 - 25 + 1) - 75)//15)
        h = torch.log(h) # (natural) log
        h = h.view(h.shape[0], -1) # flatten the non-batch dimensions
        h = self.dense(h) # (batch_size, self.n_dense_features) -> (batch_size, n_classes)
        return h


class EEGNet(nn.Module):
    def __init__(self, input_shape=(22, 1000), n_temporal_filters=4, n_spatial_filters=16, n_separable_filters=16, n_classes=4):
        super().__init__() # call __init__ method of superclass
        self.input_shape = input_shape # last two dimensions, (excluding batch size). Should be length 2.
        self.channel, self.time_points = input_shape
        self.n_temporal_filters = n_temporal_filters
        self.n_spatial_filters = n_spatial_filters
        self.n_separable_filters = n_separable_filters
        self.n_classes = n_classes
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(p=0.3)
        
        # temporal layer
        self.temporal_convolution = nn.Conv2d(1, n_temporal_filters, (1, 64), padding='same')
        self.temporal_layernorm = nn.LayerNorm([n_temporal_filters, self.channel, self.time_points])
    
        # spatial layer
        self.spatial_convolution = nn.Conv2d(n_temporal_filters, n_spatial_filters, (self.channel, 1))
        self.spatial_layernorm = nn.LayerNorm([n_spatial_filters, 1, self.time_points])
        self.temporal_average_pool_1 = nn.AvgPool2d(1, 4)

        
        # separable layer
        self.separable_convolution = nn.Conv2d(n_spatial_filters, n_separable_filters, (1, 16), padding='same')
        self.separable_layernorm = nn.LayerNorm([n_separable_filters, 1, self.time_points//4])
        self.temporal_average_pool_2 = nn.AvgPool2d((1, 8))
        
        # FC Layer
        self.dense = nn.Linear(n_separable_filters*(self.time_points//32), n_classes)

        return
    # declaring a forward method also makes the instance a callable.
    # e.g.:
    # model = EEGNet()
    # out = model(x)
    def forward(self, x):
        # x has shape (batch_size, channel, time_points)
        # Let H0 = channel, H1 = time_points
        h = x
        # this is because the torchinfo input has a weird shape
        h = h.view(-1, 1, self.channel, self.time_points) # view as (batch_size, 1, channel, time_points)
        # temporal conv 
        h = self.temporal_convolution(h) # (batch_size, 1, channel, time_points) -> (batch_size, n_temporal_filters, channel, time_points)
        # temporal layer norm
        h = self.temporal_layernorm(h) # (batch_size, n_temporal_filters, channel, time_points) -> (batch_size, n_temporal_filters, channel, time_points)
        # spacial conv
        h = self.spatial_convolution(h) # (batch_size, n_temporal_filters, channel, time_points) -> (batch_size, n_spacial_filters, 1, time_points)
        # spacial layer norm
        h = self.spatial_layernorm(h) # (batch_size, n_spacial_filters, 1, time_points) -> (batch_size, n_spacial_filters, 1, time_points)
        # ELU activation
        h = self.elu(h) # (batch_size, n_spacial_filters, 1, time_points) -> (batch_size, n_spacial_filters, 1, time_points)
        # temporal average pool 1
        h = self.temporal_average_pool_1(h) # (batch_size, n_spacial_filters, 1, time_points) -> (batch_size, n_spacial_filters, 1, time_points//4)
        # dropout
        h = self.dropout(h) # (batch_size, n_spacial_filters, 1, time_points//4) -> (batch_size, n_spacial_filters, 1, time_points//4)
        # separable conv
        h = self.separable_convolution(h) # (batch_size, n_spacial_filters, 1, time_points//4) -> (batch_size, n_separable_filteres, 1, time_points//4)
        # separable layer norm
        h = self.separable_layernorm(h) # (batch_size, n_separable_filteres, 1, time_points//4) -> (batch_size, n_separable_filteres, 1, time_points//4)
        # ELU activation
        h = self.elu(h) # (batch_size, n_separable_filteres, 1, time_points//4) -> (batch_size, n_separable_filteres, 1, time_points//4)
        # temporal average pool 2
        h = self.temporal_average_pool_2(h) # (batch_size, n_separable_filteres, 1, time_points//4) -> (batch_size, n_separable_filteres, 1, time_points//32)
        # dropout
        h = self.dropout(h) # (batch_size, n_separable_filteres, 1, time_points//32) -> (batch_size, n_separable_filteres, 1, time_points//32)
        # flatten
        h = h.view(h.shape[0], -1) # (batch_size, n_separable_filteres, 1, time_points//32) -> (batch_size, n_separable_filteres * time_points//32)
        # dense layer output
        h = self.dense(h) # (batch_size, n_separable_filteres * time_points//32) -> (batch_size, n_classes)
        return h
    
class LitModule(pl.LightningModule):
    def __init__(self, model_name, timestep_start, timestep_end):
        super().__init__()
        timestep = timestep_end-timestep_start if timestep_end-timestep_start != 0 else 1000
        print(timestep)
        if model_name == 'ShallowConvNet':
            self.model = ShallowConvNet(input_shape=(22, timestep))
        elif model_name == 'EEGNet':
            self.model = EEGNet(input_shape=(22, timestep))
        elif model_name == 'ViTransformer':
            self.model = ViTransformer(input_shape=(22, timestep))
        elif model_name == 'ATCNet':
            self.model = ATCNet(input_shape=(22, timestep))
        elif model_name == 'EEGNet_mod':
            self.model = Convolution_module(input_shape=(22, timestep))
        else:
            raise NotImplementedError
        self.learning_rate = 1e-5

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        out = self.model(x) 
        _, loss, acc = self._get_preds_loss_accuracy(out, y)

        #log for wandb
        self.log('train_loss', loss, on_step=True)
        self.log('train_accuracy', acc, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x) 
        preds, val_loss, acc = self._get_preds_loss_accuracy(out, y)

        #log for wandb
        self.log('val_loss', val_loss, on_step=True)
        self.log('val_accuracy', acc, on_step=True)

        return preds

    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x) 
        preds, _, acc = self._get_preds_loss_accuracy(out, y)

        #log for wandb
        self.log('test_accuracy', acc)

        return preds

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def _get_preds_loss_accuracy(self, logits, labels):
        preds = torch.argmax(logits, dim=1)
        loss = F.cross_entropy(logits, labels)
        accuracy = MulticlassAccuracy(num_classes=4, average='macro')
        acc = accuracy(preds, labels)
        return preds, loss, acc
