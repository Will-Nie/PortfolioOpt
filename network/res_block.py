"""
Copyright 2020 Sensetime X-lab. All Rights Reserved

Main Function:
    1. build ResBlock: you can use this classes to build residual blocks
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, kaiming_uniform_

from .activation import build_activation
from .nn_module import conv2d_block, fc_block
from .nn_module import conv2d_block2, fc_block2


class ResBlock(nn.Module):
    r'''
    Overview:
        Residual Block with 2D convolution layers, including 2 types:
            basic block:
                input channel: C
                x -> 3*3*C -> norm -> act -> 3*3*C -> norm -> act -> out
                \__________________________________________/+
            bottleneck block:
                x -> 1*1*(1/4*C) -> norm -> act -> 3*3*(1/4*C) -> norm -> act -> 1*1*C -> norm -> act -> out
                \_____________________________________________________________________________/+

    Interface:
        __init__, forward
    '''

    def __init__(self, in_channels, out_channels=None, stride=1, downsample=None, activation='relu', norm_type='LN', ):
        r"""
        Overview:
            Init the Residual Block

        Arguments:
            - in_channels (:obj:`int`): Number of channels in the input tensor
            - activation (:obj:`nn.Module`): the optional activation function
            - norm_type (:obj:`str`): type of the normalization, defalut set to batch normalization,
                                      support ['BN', 'IN', 'SyncBN', None]
            - res_type (:obj:`str`): type of residual block, support ['basic', 'bottleneck'], see overview for details
        """
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = self.in_channels if out_channels is None else out_channels
        self.activation_type = activation
        self.norm_type = norm_type
        self.stride = stride
        self.downsample = downsample
        self.conv1 = conv2d_block(in_channels=self.in_channels,
                                  out_channels=self.out_channels,
                                  kernel_size=3,
                                  stride=self.stride,
                                  padding=1,
                                  activation=self.activation_type,
                                  norm_type=self.norm_type)
        self.conv2 = conv2d_block(in_channels=self.out_channels,
                                  out_channels=self.out_channels,
                                  kernel_size=3,
                                  stride=self.stride,
                                  padding=1,
                                  activation=None,
                                  norm_type=self.norm_type)
        self.activation = build_activation(self.activation_type)

    def forward(self, x):
        r"""
        Overview:
            return the redisual block output

        Arguments:
            - x (:obj:`tensor`): the input tensor

        Returns:
            - x(:obj:`tensor`): the resblock output tensor
        """
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.activation(out)
        return out


class ResBlock2(nn.Module):
    r'''
    Overview:
        Residual Block with 2D convolution layers, including 2 types:
            basic block:
                input channel: C
                x -> 3*3*C -> norm -> act -> 3*3*C -> norm -> act -> out
                \__________________________________________/+
            bottleneck block:
                x -> 1*1*(1/4*C) -> norm -> act -> 3*3*(1/4*C) -> norm -> act -> 1*1*C -> norm -> act -> out
                \_____________________________________________________________________________/+

    Interface:
        __init__, forward
    '''

    def __init__(self, in_channels, out_channels=None, stride=1, downsample=None, activation='relu', norm_type='LN', ):
        r"""
        Overview:
            Init the Residual Block

        Arguments:
            - in_channels (:obj:`int`): Number of channels in the input tensor
            - activation (:obj:`nn.Module`): the optional activation function
            - norm_type (:obj:`str`): type of the normalization, defalut set to batch normalization,
                                      support ['BN', 'IN', 'SyncBN', None]
            - res_type (:obj:`str`): type of residual block, support ['basic', 'bottleneck'], see overview for details
        """
        super(ResBlock2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = self.in_channels if out_channels is None else out_channels
        self.activation_type = activation
        self.norm_type = norm_type
        self.stride = stride
        self.downsample = downsample
        self.conv1 = conv2d_block2(in_channels=self.in_channels,
                                   out_channels=self.out_channels,
                                   kernel_size=3,
                                   stride=self.stride,
                                   padding=1,
                                   activation=self.activation_type,
                                   norm_type=self.norm_type)
        self.conv2 = conv2d_block2(in_channels=self.out_channels,
                                   out_channels=self.out_channels,
                                   kernel_size=3,
                                   stride=self.stride,
                                   padding=1,
                                   activation=self.activation_type,
                                   norm_type=self.norm_type)
        self.activation = build_activation(self.activation_type)

    def forward(self, x):
        r"""
        Overview:
            return the redisual block output

        Arguments:
            - x (:obj:`tensor`): the input tensor

        Returns:
            - x(:obj:`tensor`): the resblock output tensor
        """
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return x


class ResFCBlock(nn.Module):
    def __init__(self, in_channels, activation='relu', norm_type=None):
        r"""
        Overview:
            Init the Residual Block

        Arguments:
            - activation (:obj:`nn.Module`): the optional activation function
            - norm_type (:obj:`str`): type of the normalization, defalut set to batch normalization
        """
        super(ResFCBlock, self).__init__()
        self.activation_type = activation
        self.norm_type = norm_type
        self.fc1 = fc_block(in_channels, in_channels, norm_type=self.norm_type, activation=self.activation_type)
        self.fc2 = fc_block(in_channels, in_channels, norm_type=self.norm_type, activation=None)
        self.activation = build_activation(self.activation_type)

    def forward(self, x):
        r"""
        Overview:
            return  output of  the residual block with 2 fully connected block

        Arguments:
            - x (:obj:`tensor`): the input tensor

        Returns:
            - x(:obj:`tensor`): the resblock output tensor
        """
        residual = x
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.activation(x + residual)
        return x


class ResFCBlock2(nn.Module):
    r'''
    Overview:
        Residual Block with 2 fully connected block
        x -> fc1 -> norm -> act -> fc2 -> norm -> act -> out
        \_____________________________________/+

    Interface:
        __init__, forward
    '''

    def __init__(self, in_channels, activation='relu', norm_type='LN'):
        r"""
        Overview:
            Init the Residual Block

        Arguments:
            - activation (:obj:`nn.Module`): the optional activation function
            - norm_type (:obj:`str`): type of the normalization, defalut set to batch normalization
        """
        super(ResFCBlock2, self).__init__()
        self.activation_type = activation
        self.fc1 = fc_block2(in_channels, in_channels, activation=self.activation_type, norm_type=norm_type)
        self.fc2 = fc_block2(in_channels, in_channels, activation=self.activation_type, norm_type=norm_type)

    def forward(self, x):
        r"""
        Overview:
            return  output of  the residual block with 2 fully connected block

        Arguments:
            - x (:obj:`tensor`): the input tensor

        Returns:
            - x(:obj:`tensor`): the resblock output tensor
        """
        residual = x
        x = self.fc1(x)
        x = self.fc2(x)
        x = x + residual
        return x


class GatedResBlock(nn.Module):
    '''
    Gated Residual Block with conv2d_block by songgl at 2020.10.23
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation=nn.ReLU(), norm_type='BN'):
        super(GatedResBlock, self).__init__()
        assert (stride == 1)
        assert (in_channels == out_channels)
        self.act = activation
        self.conv1 = conv2d_block(in_channels, out_channels, 3, 1, 1, activation=self.act, norm_type=norm_type)
        self.conv2 = conv2d_block(out_channels, out_channels, 3, 1, 1, activation=None, norm_type=norm_type)
        self.GateWeightG = nn.Sequential(
            conv2d_block(out_channels, out_channels, 1, 1, 0, activation=self.act, norm_type=None),
            conv2d_block(out_channels, out_channels, 1, 1, 0, activation=self.act, norm_type=None),
            conv2d_block(out_channels, out_channels, 1, 1, 0, activation=self.act, norm_type=None),
            conv2d_block(out_channels, out_channels, 1, 1, 0, activation=None, norm_type=None)
        )
        self.UpdateSP = nn.Parameter(torch.zeros(1))
        nn.init.constant_(self.UpdateSP, 0.1)

    def forward(self, x, NoiseMap):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.tanh(x * torch.sigmoid(self.GateWeightG(NoiseMap))) * self.UpdateSP
        x = self.act(x + residual)
        # x = x + residual
        return x


class FiLM(nn.Module):
    """
    A Feature-wise Linear Modulation Layer from
    'FiLM: Visual Reasoning with a General Conditioning Layer'
    """

    def forward(self, x, gammas, betas):
        gammas = gammas.unsqueeze(2).unsqueeze(3).expand_as(x)
        betas = betas.unsqueeze(2).unsqueeze(3).expand_as(x)
        return (gammas * x) + betas


class FiLMedResBlock(nn.Module):
    def __init__(self, in_dim, out_dim=None, with_residual=True, with_batchnorm=False,
                 with_cond=[False], dropout=0, num_extra_channels=0, extra_channel_freq=1,
                 with_input_proj=3, num_cond_maps=0, kernel_size=3, batchnorm_affine=False,
                 num_layers=1, condition_method='conv-film', debug_every=float('inf')):
        if out_dim is None:
            out_dim = in_dim
        super(FiLMedResBlock, self).__init__()
        self.with_residual = with_residual
        self.with_batchnorm = with_batchnorm
        self.with_cond = with_cond
        self.dropout = dropout
        self.extra_channel_freq = 0 if num_extra_channels == 0 else extra_channel_freq
        self.with_input_proj = with_input_proj  # Kernel size of input projection
        self.num_cond_maps = num_cond_maps
        self.kernel_size = kernel_size
        self.batchnorm_affine = batchnorm_affine
        self.num_layers = num_layers
        self.condition_method = condition_method
        self.debug_every = debug_every
        self.film = None
        self.bn1 = None
        self.drop = None

        if self.with_input_proj % 2 == 0:
            raise (NotImplementedError)
        if self.kernel_size % 2 == 0:
            raise (NotImplementedError)
        if self.num_layers >= 2:
            raise (NotImplementedError)

        if self.condition_method == 'block-input-film' and self.with_cond[0]:
            self.film = FiLM()
        if self.with_input_proj:
            self.input_proj = nn.Conv2d(in_dim + (num_extra_channels if self.extra_channel_freq >= 1 else 0),
                                        in_dim, kernel_size=self.with_input_proj, padding=self.with_input_proj // 2)

        self.conv1 = nn.Conv2d(in_dim + self.num_cond_maps +
                               (num_extra_channels if self.extra_channel_freq >= 2 else 0),
                               out_dim, kernel_size=self.kernel_size,
                               padding=self.kernel_size // 2)
        if self.condition_method == 'conv-film' and self.with_cond[0]:
            self.film = FiLM()
        if self.with_batchnorm:
            self.bn1 = nn.BatchNorm2d(out_dim, affine=((not self.with_cond[0]) or self.batchnorm_affine))
        if self.condition_method == 'bn-film' and self.with_cond[0]:
            self.film = FiLM()
        if dropout > 0:
            self.drop = nn.Dropout2d(p=self.dropout)
        if ((self.condition_method == 'relu-film' or self.condition_method == 'block-output-film')
                and self.with_cond[0]):
            self.film = FiLM()

        self.init_modules(self.modules())

    def init_modules(self, modules, init='normal'):
        if init.lower() == 'normal':
            init_params = kaiming_normal_
        elif init.lower() == 'uniform':
            init_params = kaiming_uniform_
        else:
            return
        for m in modules:
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                init_params(m.weight)

    def forward(self, x, gammas: Optional[torch.Tensor] = None, betas: Optional[torch.Tensor] = None,
                extra_channels: Optional[torch.Tensor] = None, cond_maps: Optional[torch.Tensor] = None):

        if self.film is not None:
            if self.condition_method == 'block-input-film' and self.with_cond[0]:
                x = self.film(x, gammas, betas)

        # ResBlock input projection
        if self.with_input_proj:
            if extra_channels is not None and self.extra_channel_freq >= 1:
                x = torch.cat([x, extra_channels], 1)
            x = F.relu(self.input_proj(x))
        out = x

        # ResBlock body
        if cond_maps is not None:
            out = torch.cat([out, cond_maps], 1)
        if extra_channels is not None and self.extra_channel_freq >= 2:
            out = torch.cat([out, extra_channels], 1)
        out = self.conv1(out)
        if self.film is not None:
            if self.condition_method == 'conv-film' and self.with_cond[0]:
                out = self.film(out, gammas, betas)
        if self.bn1 is not None:
            if self.with_batchnorm:
                out = self.bn1(out)
        if self.film is not None:
            if self.condition_method == 'bn-film' and self.with_cond[0]:
                out = self.film(out, gammas, betas)
        if self.drop is not None:
            if self.dropout > 0:
                out = self.drop(out)
        # out = F.relu(out)
        if self.film is not None:
            if self.condition_method == 'relu-film' and self.with_cond[0]:
                out = self.film(out, gammas, betas)

        # ResBlock remainder
        if self.with_residual:
            out = F.relu(x + out)
        if self.film is not None:
            if self.condition_method == 'block-output-film' and self.with_cond[0]:
                out = self.film(out, gammas, betas)
        return out
