# 2022.03.16-Changed for building NexToU
#            Harbin Institute of Technology (Shenzhen), <pcshi@stu.hit.edu.cn>

#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import torch
from copy import deepcopy
from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn
import numpy as np
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
import torch.nn.functional

from nnunet.network_architecture.torch_nn import BasicConv, batched_index_select, act_layer
from nnunet.network_architecture.torch_edge import DenseDilatedKnnGraph
from nnunet.network_architecture.pos_embed import get_2d_relative_pos_embed, get_3d_relative_pos_embed
import torch.nn.functional as F
from timm.models.layers import DropPath
from einops import rearrange

class ConvDropoutNormNonlin(nn.Module):
    """
    fixes a bug in ConvDropoutNormNonlin where lrelu was used regardless of nonlin. Bad.
    """

    def __init__(self, input_channels, output_channels,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None):
        super(ConvDropoutNormNonlin, self).__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        self.conv = self.conv_op(input_channels, output_channels, **self.conv_kwargs)
        if self.dropout_op is not None and self.dropout_op_kwargs['p'] is not None and self.dropout_op_kwargs[
            'p'] > 0:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
        else:
            self.dropout = None
        self.instnorm = self.norm_op(output_channels, **self.norm_op_kwargs)
        self.lrelu = self.nonlin(**self.nonlin_kwargs)

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.lrelu(self.instnorm(x))


class ConvDropoutNonlinNorm(ConvDropoutNormNonlin):
    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.instnorm(self.lrelu(x))


class StackedConvLayers(nn.Module):
    def __init__(self, input_feature_channels, output_feature_channels, num_convs,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, first_stride=None, basic_block=ConvDropoutNormNonlin):
        '''
        stacks ConvDropoutNormLReLU layers. initial_stride will only be applied to first layer in the stack. The other parameters affect all layers
        :param input_feature_channels:
        :param output_feature_channels:
        :param num_convs:
        :param dilation:
        :param kernel_size:
        :param padding:
        :param dropout:
        :param initial_stride:
        :param conv_op:
        :param norm_op:
        :param dropout_op:
        :param inplace:
        :param neg_slope:
        :param norm_affine:
        :param conv_bias:
        '''
        self.input_channels = input_feature_channels
        self.output_channels = output_feature_channels

        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        if first_stride is not None:
            self.conv_kwargs_first_conv = deepcopy(conv_kwargs)
            self.conv_kwargs_first_conv['stride'] = first_stride
        else:
            self.conv_kwargs_first_conv = conv_kwargs

        super(StackedConvLayers, self).__init__()
        self.blocks = nn.Sequential(
            *([basic_block(input_feature_channels, output_feature_channels, self.conv_op,
                           self.conv_kwargs_first_conv,
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs)] +
              [basic_block(output_feature_channels, output_feature_channels, self.conv_op,
                           self.conv_kwargs,
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs) for _ in range(num_convs - 1)]))

    def forward(self, x):
        return self.blocks(x)


def print_module_training_status(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d) or isinstance(module, nn.Dropout3d) or \
            isinstance(module, nn.Dropout2d) or isinstance(module, nn.Dropout) or isinstance(module, nn.InstanceNorm3d) \
            or isinstance(module, nn.InstanceNorm2d) or isinstance(module, nn.InstanceNorm1d) \
            or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d) or isinstance(module,
                                                                                                      nn.BatchNorm1d):
        print(str(module), module.training)


class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Upsample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                                         align_corners=self.align_corners)


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0, conv_op=nn.Conv3d, norm_op=nn.BatchNorm3d, norm_op_kwargs=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            conv_op(in_features, hidden_features, 1, stride=1, padding=0),
            norm_op(hidden_features, **norm_op_kwargs),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            conv_op(hidden_features, out_features, 1, stride=1, padding=0),
            norm_op(out_features, **norm_op_kwargs),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x  # .reshape(B, C, N, 1)

class MRConv(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True, conv_op=nn.Conv3d, dropout_op=nn.Dropout3d):
        super(MRConv, self).__init__()
        self.conv_op = conv_op
        self.nn = BasicConv([in_channels*2, out_channels], act=act, norm=norm, bias=bias, drop=0., conv_op=conv_op, dropout_op=dropout_op)

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1], self.conv_op)
        if y is not None:
            x_j = batched_index_select(y, edge_index[0], self.conv_op)
        else:
            x_j = batched_index_select(x, edge_index[0], self.conv_op)
        x_j, _ = torch.max(x_j - x_i, -1, keepdim=True)
        b, c, n, _ = x.shape
        x = torch.cat([x.unsqueeze(2), x_j.unsqueeze(2)], dim=2).reshape(b, 2 * c, n, _)
       
        if self.conv_op == nn.Conv2d:
            pass
        elif self.conv_op == nn.Conv3d:
            x = torch.unsqueeze(x, dim=4) 
        else:
            raise NotImplementedError('conv operation [%s] is not found' % self.conv_op)
        
        return self.nn(x)

class GraphConv(nn.Module):
    """
    Static graph convolution layer
    """
    def __init__(self, in_channels, out_channels, conv='edge', act='relu', norm=None, bias=True, conv_op=nn.Conv3d, dropout_op=nn.Dropout3d):
        super(GraphConv, self).__init__()
        if conv == 'mr':
            self.gconv = MRConv(in_channels, out_channels, act, norm, bias, conv_op, dropout_op)
        else:
            raise NotImplementedError('conv:{} is not supported'.format(conv))

    def forward(self, x, edge_index, y=None):
        return self.gconv(x, edge_index, y)


class DyGraphConv(GraphConv):
    """
    Dynamic graph convolution layer
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv='edge', act='relu',
                 norm=None, bias=True, stochastic=False, epsilon=0.0, r=1, conv_op=nn.Conv3d, dropout_op=nn.Dropout3d):
        super(DyGraphConv, self).__init__(in_channels, out_channels, conv, act, norm, bias, conv_op, dropout_op)
        self.k = kernel_size
        self.d = dilation
        self.r = r
        self.dilated_knn_graph = DenseDilatedKnnGraph(kernel_size, dilation, stochastic, epsilon)
        self.conv_op = conv_op
        self.dropout_op = dropout_op
        if self.conv_op == nn.Conv2d:
            self.avg_pool = F.avg_pool2d
        elif self.conv_op == nn.Conv3d:
            self.avg_pool = F.avg_pool3d
        else:
            raise NotImplementedError('conv operation [%s] is not found' % self.conv_op)

    def forward(self, x, relative_pos=None):
        if self.conv_op == nn.Conv2d:
            B, C, H, W = x.shape
        elif self.conv_op == nn.Conv3d:
            B, C, H, W, D = x.shape
        else:
            raise NotImplementedError('conv operation [%s] is not found' % self.conv_op)

        y = None
        if self.r > 1:
            y = self.avg_pool(x, self.r, self.r)
            y = y.reshape(B, C, -1, 1).contiguous()            
        x = x.reshape(B, C, -1, 1).contiguous()
        edge_index = self.dilated_knn_graph(x, y, relative_pos)
        x = super(DyGraphConv, self).forward(x, edge_index, y)
        if self.conv_op == nn.Conv2d:
            return x.reshape(B, -1, H, W).contiguous()
        elif self.conv_op == nn.Conv3d:
            return x.reshape(B, -1, H, W, D).contiguous()
        else:
            raise NotImplementedError('conv operation [%s] is not found' % self.conv_op)


class PoolDyGraphConv(GraphConv):
    """
    Dynamic graph convolution layer
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv='edge', act='relu',
                 norm=None, bias=True, stochastic=False, epsilon=0.0, r=1, conv_op=nn.Conv3d, dropout_op=nn.Dropout3d, img_shape=None, img_min_shape=None):
        super(PoolDyGraphConv, self).__init__(in_channels, out_channels, conv, act, norm, bias, conv_op, dropout_op)
        self.k = kernel_size
        self.d = dilation
        self.r = r
        self.dilated_knn_graph = DenseDilatedKnnGraph(kernel_size, dilation, stochastic, epsilon)
        self.conv_op = conv_op
        self.dropout_op = dropout_op

        n = 1
        for h in img_shape:
            n = n * h
        
        n_small = 1
        for h_small in img_min_shape:
            n_small = n_small * h_small * 4

        if n > n_small:
            pool_size = [2 if h % 2 == 0 else 1 for h in img_shape]
        else:
            pool_size = [1 for h in img_shape]

        self.pool_size = pool_size
        
        if self.conv_op == nn.Conv2d:
            self.avg_pool = F.avg_pool2d
            self.max_pool_input = nn.MaxPool2d(pool_size, stride=pool_size, return_indices=True)
            self.max_unpool_output = nn.MaxUnpool2d(pool_size, stride=pool_size)
        elif self.conv_op == nn.Conv3d:
            self.avg_pool = F.avg_pool3d
            self.max_pool_input = nn.MaxPool3d(pool_size, stride=pool_size, return_indices=True)
            self.max_unpool_output = nn.MaxUnpool3d(pool_size, stride=pool_size)
        else:
            raise NotImplementedError('conv operation [%s] is not found' % self.conv_op)

    def forward(self, x, relative_pos=None):
        if self.conv_op == nn.Conv2d:
            B, C, H, W = x.shape
        elif self.conv_op == nn.Conv3d:
            B, C, S, H, W = x.shape
        else:
            raise NotImplementedError('conv operation [%s] is not found' % self.conv_op)

        x, indices = self.max_pool_input(x)   
        y = None
        if self.r > 1:
            y = self.avg_pool(x, self.r, self.r)
            y = y.reshape(B, C, -1, 1).contiguous()   

        x = x.reshape(B, C, -1, 1).contiguous()
        indices = indices.reshape(B, C, -1, 1).contiguous()

        edge_index = self.dilated_knn_graph(x, y, relative_pos)
        x = super(PoolDyGraphConv, self).forward(x, edge_index, y)
        
        indices_cat = torch.cat((indices, indices), 1)
        
        if self.conv_op == nn.Conv2d:
            H_pool, W_pool = H // self.pool_size[0], W // self.pool_size[1]
            x = x.reshape(B, -1, H_pool, W_pool).contiguous()
            indices_cat = indices_cat.reshape(B, -1, H_pool, W_pool).contiguous()
        elif self.conv_op == nn.Conv3d:
            S_pool, H_pool, W_pool = S // self.pool_size[0], H // self.pool_size[1], W // self.pool_size[2]
            x = x.reshape(B, -1, S_pool, H_pool, W_pool).contiguous()
            indices_cat = indices_cat.reshape(B, -1, S_pool, H_pool, W_pool).contiguous()
        else:
            raise NotImplementedError('conv operation [%s] is not found' % self.conv_op)
        
        x = self.max_unpool_output(x, indices_cat)

        return x
        
class Grapher(nn.Module):
    """
    Grapher module with graph convolution and fc layers
    """
    def __init__(self, in_channels, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
                 bias=True,  stochastic=False, epsilon=0.0, r=1, n=196, drop_path=0.0, relative_pos=False, 
                 conv_op=nn.Conv3d, norm_op=nn.BatchNorm3d, dropout_op=nn.Dropout3d):
        super(Grapher, self).__init__()
        self.channels = in_channels
        self.n = n
        self.r = r
        self.conv_op = conv_op
        self.fc1 = nn.Sequential(
            conv_op(in_channels, in_channels, 1, stride=1, padding=0),
            norm_op(in_channels),
        )
        self.graph_conv = DyGraphConv(in_channels, in_channels * 2, kernel_size, dilation, conv,
                                      act, norm, bias, stochastic, epsilon, r, conv_op, dropout_op)
        self.fc2 = nn.Sequential(
            conv_op(in_channels * 2, in_channels, 1, stride=1, padding=0),
            norm_op(in_channels),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.relative_pos = None
        if relative_pos:
            
            if self.conv_op == nn.Conv2d:
                relative_pos_tensor = torch.from_numpy(np.float32(get_2d_relative_pos_embed(in_channels,
                int(n**(1/2))))).unsqueeze(0).unsqueeze(1)
                relative_pos_tensor = F.interpolate(
                        relative_pos_tensor, size=(n, n//(r*r)), mode='bicubic', align_corners=False)
            elif self.conv_op == nn.Conv3d:
                relative_pos_tensor = torch.from_numpy(np.float32(get_3d_relative_pos_embed(in_channels,
                int(n**(1/3))))).unsqueeze(0).unsqueeze(1) 
                relative_pos_tensor = F.interpolate(
                        relative_pos_tensor, size=(n, n//(r*r*r)), mode='bicubic', align_corners=False)
            else:
                raise NotImplementedError('conv operation [%s] is not found' % self.conv_op)
            self.relative_pos = nn.Parameter(-relative_pos_tensor.squeeze(1), requires_grad=False)

    def _get_relative_pos(self, relative_pos, size_tuple):
        if self.conv_op == nn.Conv2d:
            H, W = size_tuple
            if relative_pos is None or H * W == self.n:
                return relative_pos
            else:
                N = H * W
                N_reduced = N // (self.r * self.r)
                return F.interpolate(relative_pos.unsqueeze(0), size=(N, N_reduced), mode="bicubic").squeeze(0)

        elif self.conv_op == nn.Conv3d:
            H, W, D = size_tuple
            if relative_pos is None or H * W * D == self.n:
                return relative_pos
            else:
                N = H * W * D
                N_reduced = N // (self.r * self.r * self.r)
                return F.interpolate(relative_pos.unsqueeze(0), size=(N, N_reduced), mode="bicubic").squeeze(0)
        else:
            raise NotImplementedError('conv operation [%s] is not found' % self.conv_op)

        
    def forward(self, x):
        _tmp = x
        x = self.fc1(x)
        if self.conv_op == nn.Conv2d:
            B, C, H, W = x.shape
            size_tuple = (H, W)
            relative_pos = self._get_relative_pos(self.relative_pos, size_tuple)
        elif self.conv_op == nn.Conv3d:
            B, C, H, W, D = x.shape
            size_tuple = (H, W, D)
            relative_pos = self._get_relative_pos(self.relative_pos, size_tuple)
        else:
            raise NotImplementedError('conv operation [%s] is not found' % self.conv_op)
        
        x = self.graph_conv(x, relative_pos)
        x = self.fc2(x)
        x = self.drop_path(x) + _tmp
        return x

def window_partition(x, window_size):
    """
    Args:
        x: (B, C, S, H, W) or (B, C, H, W)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, window_size, C)
    """
    
    if len(x.shape) == 4:
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        windows = rearrange(x, 'b (h p1) (w p2) c -> (b h w) p1 p2 c',
                            p1=window_size[0], p2=window_size[1], c=C)
        windows = windows.permute(0, 3, 1, 2)

    elif len(x.shape) == 5:
        B, C, S, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1)
        windows = rearrange(x, 'b (s p1) (h p2) (w p3) c -> (b s h w) p1 p2 p3 c',
                            p1=window_size[0], p2=window_size[1], p3=window_size[2], c=C)
        windows = windows.permute(0, 4, 1, 2, 3)
    else:
        raise NotImplementedError('len(x.shape) [%d] is equal to 4 or 5' % len(x.shape))
    
    return windows

def window_reverse(windows, window_size, size_tuple):
    """
    Args:
        windows: (num_windows*B, C, window_size, window_size, window_size)
        window_size (int): Window size
        S (int): Slice of image
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, C, S ,H, W)
    """
    if len(windows.shape) == 4:
        H, W = size_tuple
        B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
        windows = windows.permute(0, 2, 3, 1)
        x = rearrange(windows, '(b h w) p1 p2 c -> b (h p1) (w p2) c',
                    p1=window_size[0], p2=window_size[1], b=B, h=H//window_size[0], w=W//window_size[1])
        x = x.permute(0, 3, 1, 2)

    elif len(windows.shape) == 5:
        S, H, W = size_tuple
        B = int(windows.shape[0] / (S * H * W / window_size[0] / window_size[1] / window_size[2]))        
        windows = windows.permute(0, 2, 3, 4, 1)
        x = rearrange(windows, '(b s h w) p1 p2 p3 c -> b (s p1) (h p2) (w p3) c',
                    p1=window_size[0], p2=window_size[1], p3=window_size[2], b=B,
                    s=S//window_size[0], h=H//window_size[1], w=W//window_size[2])
        x = x.permute(0, 4, 1, 2, 3)
    else:
        raise NotImplementedError('len(x.shape) [%d] is equal to 4 or 5' % len(x.shape))

    return x

class SwinGrapher(nn.Module):
    """
    SwinGrapher module with graph convolution and fc layers
    """
    def __init__(self, in_channels, img_shape, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
                 bias=True,  stochastic=False, epsilon=0.0, r=1, n=196, drop_path=0.0, relative_pos=False, 
                 conv_op=nn.Conv3d, norm_op=nn.BatchNorm3d, norm_op_kwargs=None, dropout_op=nn.Dropout3d, window_size=[3, 6, 6], shift_size=[0, 0, 0]):
        super(SwinGrapher, self).__init__()
        self.channels = in_channels
        # self.n = n
        self.r = r
        self.conv_op = conv_op
        self.img_shape = img_shape
        self.window_size = window_size
        self.shift_size = shift_size

        self.fc1 = nn.Sequential(
            conv_op(in_channels, in_channels, 1, stride=1, padding=0),
            norm_op(in_channels, **norm_op_kwargs),
        )
        norm = 'batch'
        self.graph_conv = DyGraphConv(in_channels, in_channels * 2, kernel_size, dilation, conv,
                                      act, norm, bias, stochastic, epsilon, r, conv_op, dropout_op)
        self.fc2 = nn.Sequential(
            conv_op(in_channels * 2, in_channels, 1, stride=1, padding=0),
            norm_op(in_channels, **norm_op_kwargs),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        n = 1
        for h in self.window_size:
            n = n * h
        self.n = n
        self.relative_pos = None
        if relative_pos:
            
            if self.conv_op == nn.Conv2d:
                relative_pos_tensor = torch.from_numpy(np.float32(get_2d_relative_pos_embed(in_channels,
                int(n**(1/2))))).unsqueeze(0).unsqueeze(1) ####
                relative_pos_tensor = F.interpolate(
                        relative_pos_tensor, size=(n, n//(r*r)), mode='bicubic', align_corners=False)
            elif self.conv_op == nn.Conv3d:
                relative_pos_tensor = torch.from_numpy(np.float32(get_3d_relative_pos_embed(in_channels,
                int(n**(1/3))))).unsqueeze(0).unsqueeze(1) ####
                relative_pos_tensor = F.interpolate(
                        relative_pos_tensor, size=(n, n//(r*r*r)), mode='bicubic', align_corners=False)
            else:
                raise NotImplementedError('conv operation [%s] is not found' % self.conv_op)
            self.relative_pos = nn.Parameter(-relative_pos_tensor.squeeze(1), requires_grad=False)

    def _get_relative_pos(self, relative_pos, window_size_tuple):
        if self.conv_op == nn.Conv2d:
            H, W = window_size_tuple
            if relative_pos is None or H * W == self.n:
                return relative_pos
            else:
                N = H * W
                N_reduced = N // (self.r * self.r)
                return F.interpolate(relative_pos.unsqueeze(0), size=(N, N_reduced), mode="bicubic").squeeze(0)

        elif self.conv_op == nn.Conv3d:
            S, H, W = window_size_tuple
            if relative_pos is None or S * H * W == self.n:
                return relative_pos
            else:
                N = S * H * W
                N_reduced = N // (self.r * self.r * self.r)
                return F.interpolate(relative_pos.unsqueeze(0), size=(N, N_reduced), mode="bicubic").squeeze(0)
        else:
            raise NotImplementedError('conv operation [%s] is not found' % self.conv_op)

        
    def forward(self, x):
        _tmp = x
        if self.conv_op == nn.Conv2d:
            B, C, H, W = x.shape
            size_tuple = (H, W)
            h, w = self.img_shape
            assert H == h and W == w, "input feature has wrong size"
        elif self.conv_op == nn.Conv3d:
            B, C, S, H, W = x.shape
            size_tuple = (S, H, W)
            s, h, w = self.img_shape
            assert S == s and H == h and W == w, "input feature has wrong size"
        else:
            raise NotImplementedError('conv operation [%s] is not found' % self.conv_op)

        if max(self.shift_size) > 0 and self.conv_op == nn.Conv2d:
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(2, 3))
        elif max(self.shift_size) > 0 and self.conv_op == nn.Conv3d:
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(2, 3, 4))
        else:
            shifted_x = x

        # partition windows
        # nW*B, C, window_size, window_size, window_size
        x_windows = window_partition(shifted_x, self.window_size)

        x = self.fc1(x_windows)
        if self.conv_op == nn.Conv2d:
            B_, C, H, W = x.shape
            window_size_tuple = (H, W)
            relative_pos = self._get_relative_pos(self.relative_pos, window_size_tuple)
        elif self.conv_op == nn.Conv3d:
            B_, C, S, H, W = x.shape
            window_size_tuple = (S, H, W)
            relative_pos = self._get_relative_pos(self.relative_pos, window_size_tuple)
        else:
            raise NotImplementedError('conv operation [%s] is not found' % self.conv_op)
        
        x = self.graph_conv(x, relative_pos)
        gnn_windows = self.fc2(x)

        shifted_x = window_reverse(gnn_windows, self.window_size, size_tuple)

        # reverse cyclic shift
        if max(self.shift_size) > 0 and self.conv_op == nn.Conv2d:
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1]), dims=(2, 3))
        elif max(self.shift_size) > 0 and self.conv_op == nn.Conv3d:
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(2, 3, 4))
        else:
            x = shifted_x

        x = self.drop_path(x) + _tmp
        return x

class PoolGrapher(nn.Module):
    """
    PoolGrapher module with graph convolution and fc layers
    """
    def __init__(self, in_channels, img_shape, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
                 bias=True,  stochastic=False, epsilon=0.0, r=1, n=196, drop_path=0.0, relative_pos=False, 
                 conv_op=nn.Conv3d, norm_op=nn.BatchNorm3d, norm_op_kwargs=None, dropout_op=nn.Dropout3d, img_min_shape=None):
        super(PoolGrapher, self).__init__()
        self.channels = in_channels
        # self.n = n
        self.r = r
        self.conv_op = conv_op
        self.img_shape = img_shape
        

        self.fc1 = nn.Sequential(
            conv_op(in_channels, in_channels, 1, stride=1, padding=0),
            norm_op(in_channels, **norm_op_kwargs),
        )
        self.graph_conv = PoolDyGraphConv(in_channels, in_channels * 2, kernel_size, dilation, conv,
                                      act, norm, bias, stochastic, epsilon, r, conv_op, dropout_op, img_shape=img_shape, img_min_shape=img_min_shape)
        self.fc2 = nn.Sequential(
            conv_op(in_channels * 2, in_channels, 1, stride=1, padding=0),
            norm_op(in_channels, **norm_op_kwargs),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        n = 1
        for h in img_shape:
            n = n * h
        
        n_small = 1
        for h_small in img_min_shape:
            n_small = n_small * h_small * 4

        if n > n_small:
            pool_size = [2 if h % 2 == 0 else 1 for h in img_shape]
        else:
            pool_size = [1 for h in img_shape]
        
        self.pool_size = pool_size
        
        p_num = 1
        for p in pool_size:
            p_num = p_num * p

        n = n // p_num
        self.n = n
        self.relative_pos = None
        if relative_pos:
            if self.conv_op == nn.Conv2d:
                relative_pos_tensor = torch.from_numpy(np.float32(get_2d_relative_pos_embed(in_channels,
                int(n**(1/2))))).unsqueeze(0).unsqueeze(1) 
                relative_pos_tensor = F.interpolate(
                        relative_pos_tensor, size=(n, n//(r*r)), mode='bicubic', align_corners=False)
            elif self.conv_op == nn.Conv3d:
                relative_pos_tensor = torch.from_numpy(np.float32(get_3d_relative_pos_embed(in_channels,
                int(n**(1/3))))).unsqueeze(0).unsqueeze(1) 
                relative_pos_tensor = F.interpolate(
                        relative_pos_tensor, size=(n, n//(r*r*r)), mode='bicubic', align_corners=False)
            else:
                raise NotImplementedError('conv operation [%s] is not found' % self.conv_op)
            self.relative_pos = nn.Parameter(-relative_pos_tensor.squeeze(1), requires_grad=False)

    def _get_relative_pos(self, relative_pos, size_tuple):
        if self.conv_op == nn.Conv2d:
            H, W = size_tuple
            if relative_pos is None or H * W == self.n:
                return relative_pos
            else:
                N = H * W
                N_reduced = N // (self.r * self.r)
                return F.interpolate(relative_pos.unsqueeze(0), size=(N, N_reduced), mode="bicubic").squeeze(0)

        elif self.conv_op == nn.Conv3d:
            S, H, W = size_tuple
            if relative_pos is None or S * H * W == self.n:
                return relative_pos
            else:
                N = S * H * W
                N_reduced = N // (self.r * self.r * self.r)
                return F.interpolate(relative_pos.unsqueeze(0), size=(N, N_reduced), mode="bicubic").squeeze(0)
        else:
            raise NotImplementedError('conv operation [%s] is not found' % self.conv_op)

        
    def forward(self, x):
        _tmp = x
        if self.conv_op == nn.Conv2d:
            B, C, H, W = x.shape
            size_tuple = (H, W)
            h, w = self.img_shape
            assert H == h and W == w, "input feature has wrong size"
        elif self.conv_op == nn.Conv3d:
            B, C, S, H, W = x.shape
            size_tuple = (S, H, W)
            s, h, w = self.img_shape
            assert S == s and H == h and W == w, "input feature has wrong size"
        else:
            raise NotImplementedError('conv operation [%s] is not found' % self.conv_op)

        x = self.fc1(x)
        if self.conv_op == nn.Conv2d:
            B_, C, H, W = x.shape
            size_tuple = (H // self.pool_size[0], W // self.pool_size[1])
            relative_pos = self._get_relative_pos(self.relative_pos, size_tuple)
        elif self.conv_op == nn.Conv3d:
            B_, C, S, H, W = x.shape
            size_tuple = (S // self.pool_size[0], H // self.pool_size[1], W // self.pool_size[2])
            relative_pos = self._get_relative_pos(self.relative_pos, size_tuple)
        else:
            raise NotImplementedError('conv operation [%s] is not found' % self.conv_op)
        
        x = self.graph_conv(x, relative_pos)
        x = self.fc2(x)

        x = self.drop_path(x) + _tmp
        return x

class Efficient_ViG_blocks(nn.Module):
    def __init__(self, channels, img_shape, index, conv_layer_d_num, opt=None, conv_op=nn.Conv3d, norm_op=nn.BatchNorm3d, norm_op_kwargs=None,
                    dropout_op=nn.Dropout3d, **kwargs):
        super(Efficient_ViG_blocks, self).__init__()

        blocks = []
        k = opt.k
        conv = opt.conv
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        epsilon = opt.epsilon
        stochastic = opt.use_stochastic
        drop_path = opt.drop_path
        reduce_ratios = opt.reduce_ratios 
        blocks_num_list = opt.blocks
        n_size_list = opt.n_size_list
        img_min_shape = opt.img_min_shape

        self.n_blocks = sum(blocks_num_list)        
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]
        sum_blocks = sum(blocks_num_list[conv_layer_d_num-2:index])
        idx_list = [(k+sum_blocks) for k in range(0, blocks_num_list[index])]
        
        if conv_op == nn.Conv2d:
            H_min, W_min = img_min_shape
            max_dilation = (H_min * W_min) // max(k)
            window_size = img_min_shape
            window_size_n = window_size[0] * window_size[1]   
        elif conv_op == nn.Conv3d:
            H_min, W_min, D_min = img_min_shape
            max_dilation = (H_min * W_min * D_min) // max(k)  
            window_size = img_min_shape
            window_size_n = window_size[0] * window_size[1] * window_size[2]      
        else:
            raise NotImplementedError('conv operation [%s] is not found' % conv_op)

        i = conv_layer_d_num-2 + index
        for j in range(blocks_num_list[index]):
            idx = idx_list[j]
            if conv_op == nn.Conv2d:
                shift_size = [window_size[0] // 2, window_size[1] // 2]
            elif conv_op == nn.Conv3d:
                shift_size = [window_size[0] // 2, window_size[1] // 2, window_size[2] // 2]
            else:
                raise NotImplementedError('conv operation [%s] is not found' % conv_op)

            blocks.append(nn.Sequential(PoolGrapher(channels, img_shape, k[i], min(idx // 4 + 1, max_dilation), conv, act, norm,
                    bias, stochastic, epsilon, reduce_ratios[i], n=n_size_list[i+2], drop_path=dpr[idx],
                    relative_pos=True, conv_op=conv_op, norm_op=norm_op, norm_op_kwargs=norm_op_kwargs, dropout_op=dropout_op, img_min_shape=img_min_shape), 
                    FFN(channels, channels * 4, act=act, drop_path=dpr[idx], conv_op=conv_op, norm_op=norm_op, norm_op_kwargs=norm_op_kwargs),
                    SwinGrapher(channels, img_shape, k[i], min(idx // 4 + 1, max_dilation), conv, act, norm,
                    bias, stochastic, epsilon, 1, n=window_size_n, drop_path=dpr[idx],
                    relative_pos=True, conv_op=conv_op, norm_op=norm_op, norm_op_kwargs=norm_op_kwargs, dropout_op=dropout_op, 
                    window_size=window_size, shift_size=shift_size), 
                    FFN(channels, channels * 4, act=act, drop_path=dpr[idx], conv_op=conv_op, norm_op=norm_op, norm_op_kwargs=norm_op_kwargs)))

        blocks = nn.Sequential(*blocks)
        self.blocks = blocks

    def forward(self, x): 
        x = self.blocks(x)
        return x

class final_patch_expanding(nn.Module):
    def __init__(self, dim, num_class, expand_patch_size=None, conv_op=None):
        super().__init__()

        if conv_op == nn.Conv2d:
            self.up = nn.ConvTranspose2d(dim, num_class, expand_patch_size, expand_patch_size)
        elif conv_op == nn.Conv3d:
            self.up = nn.ConvTranspose3d(dim, num_class, expand_patch_size, expand_patch_size)
        else:
            raise NotImplementedError('conv operation [%s] is not found' % conv_op)
      
    def forward(self,x):
        x = self.up(x)
        return x 

class NexToU(SegmentationNetwork):
    DEFAULT_BATCH_SIZE_3D = 2
    DEFAULT_PATCH_SIZE_3D = (64, 192, 160)
    SPACING_FACTOR_BETWEEN_STAGES = 2
    BASE_NUM_FEATURES_3D = 30
    MAX_NUMPOOL_3D = 999
    MAX_NUM_FILTERS_3D = 312

    DEFAULT_PATCH_SIZE_2D = (256, 256)
    BASE_NUM_FEATURES_2D = 30
    DEFAULT_BATCH_SIZE_2D = 50
    MAX_NUMPOOL_2D = 999
    MAX_FILTERS_2D = 480 #384

    use_this_for_batch_size_computation_2D = 19739648
    use_this_for_batch_size_computation_3D = 520000000  # 505789440

    def __init__(self, input_channels, base_num_features, num_classes, num_pool, patch_size, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True, dropout_in_localization=False,
                 final_nonlin=softmax_helper, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None, basic_block=ConvDropoutNormNonlin,
                 seg_output_use_bias=False, opt=None):
        """
        basically more flexible than v1, architecture is the same

        Does this look complicated? Nah bro. Functionality > usability

        This does everything you need, including world peace.

        Questions? -> f.isensee@dkfz.de
        """
        super(NexToU, self).__init__()
        self.convolutional_upsampling = convolutional_upsampling
        self.convolutional_pooling = convolutional_pooling
        self.upscale_logits = upscale_logits
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}

        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}

        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.weightInitializer = weightInitializer
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.num_classes = num_classes
        self.final_nonlin = final_nonlin
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision

        if conv_op == nn.Conv2d:
            upsample_mode = 'bilinear'
            pool_op = nn.MaxPool2d
            transpconv = nn.ConvTranspose2d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3)] * (num_pool + 1)
        elif conv_op == nn.Conv3d:
            upsample_mode = 'trilinear'
            pool_op = nn.MaxPool3d
            transpconv = nn.ConvTranspose3d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3, 3)] * (num_pool + 1)
        else:
            raise ValueError("unknown convolution dimensionality, conv op: %s" % str(conv_op))

        self.input_shape_must_be_divisible_by = np.prod(pool_op_kernel_sizes, 0, dtype=np.int64)
        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes

        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])

        if max_num_features is None:
            if self.conv_op == nn.Conv3d:
                self.max_num_features = self.MAX_NUM_FILTERS_3D
            else:
                self.max_num_features = self.MAX_FILTERS_2D
        else:
            self.max_num_features = max_num_features

        self.conv_blocks_context = []
        self.conv_blocks_localization = []
        self.td = []
        self.tu = []
        self.seg_outputs = []

        output_features = base_num_features
        input_features = input_channels

        img_shape_list = []
        n_size_list = []

        if conv_op == nn.Conv2d:
            h, w = patch_size[0], patch_size[1]
            img_shape_list.append((h, w))
            n_size_list.append(h * w)

            for i in range(len(pool_op_kernel_sizes)):
                h_k, w_k = pool_op_kernel_sizes[i]
                h //= h_k
                w //= w_k
                img_shape_list.append((h, w))
                n_size_list.append(h * w)

        elif conv_op == nn.Conv3d:
            h, w, d = patch_size[0], patch_size[1], patch_size[2]
            img_shape_list.append((h, w, d))
            n_size_list.append(h * w * d)

            for i in range(len(pool_op_kernel_sizes)):
                h_k, w_k, d_k = pool_op_kernel_sizes[i]
                h //= h_k
                w //= w_k
                d //= d_k
                img_shape_list.append((h, w, d))
                n_size_list.append(h * w * d)
        else:
            raise ValueError("unknown convolution dimensionality, conv op: %s" % str(conv_op))

        img_min_shape = img_shape_list[-1]

        self.opt = opt
        self.opt.img_min_shape = img_min_shape

        conv_layer_d_num = 2
        self.conv_layer_d_num = conv_layer_d_num
        if self.conv_op == nn.Conv2d:
            H, W = img_shape_list[conv_layer_d_num-1]
            channels_num = min(base_num_features*(2**(conv_layer_d_num-1)), self.max_num_features)
            self.pos_embed = nn.Parameter(torch.zeros(1, channels_num, H, W)) #224//4, 224//4
        elif self.conv_op == nn.Conv3d:
            H, W, D = img_shape_list[conv_layer_d_num-1]
            channels_num = min(base_num_features*(2**(conv_layer_d_num-1)), self.max_num_features)
            self.pos_embed = nn.Parameter(torch.zeros(1, channels_num, H, W, D))
        else:
            raise NotImplementedError('conv operation [%s] is not found' % self.conv_op)
                
        self.opt.n_size_list = n_size_list

        d_output_channels_list = []
        for d in range(num_pool):
            # determine the first stride
            if d != 0 and self.convolutional_pooling:
                first_stride = pool_op_kernel_sizes[d - 1]
            else:
                first_stride = None

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[d]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[d]
            

            if d < conv_layer_d_num:
                # add convolutions
                self.conv_blocks_context.append(StackedConvLayers(input_features, output_features, num_conv_per_stage,
                                                                self.conv_op, self.conv_kwargs, self.norm_op,
                                                                self.norm_op_kwargs, self.dropout_op,
                                                                self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                                                first_stride, basic_block=basic_block))
            else:
                self.conv_blocks_context.append(nn.Sequential(
                                            StackedConvLayers(input_features, output_features, num_conv_per_stage - 1, self.conv_op, self.conv_kwargs,
                                                            self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                                                            self.nonlin_kwargs, first_stride, basic_block=basic_block),
                                            Efficient_ViG_blocks(output_features, img_shape_list[d], d-conv_layer_d_num, conv_layer_d_num, opt=self.opt, conv_op=self.conv_op,
                                                                                norm_op=self.norm_op, norm_op_kwargs=self.norm_op_kwargs, dropout_op=self.dropout_op)))
            d_output_channels_list.append(output_features)

            if not self.convolutional_pooling:
                self.td.append(pool_op(pool_op_kernel_sizes[d]))
            input_features = output_features
            output_features = int(np.round(output_features * feat_map_mul_on_downscale))

            output_features = min(output_features, self.max_num_features)

        # now the bottleneck.
        # determine the first stride
        if self.convolutional_pooling:
            first_stride = pool_op_kernel_sizes[-1]
        else:
            first_stride = None

        # the output of the last conv must match the number of features from the skip connection if we are not using
        # convolutional upsampling. If we use convolutional upsampling then the reduction in feature maps will be
        # done by the transposed conv
        if self.convolutional_upsampling:
            final_num_features = output_features
        else:
            final_num_features = d_output_channels_list[-1] #self.conv_blocks_context[-1].output_channels

        self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[num_pool]
        self.conv_kwargs['padding'] = self.conv_pad_sizes[num_pool]
        self.conv_blocks_context.append(nn.Sequential(
            StackedConvLayers(input_features, output_features, num_conv_per_stage - 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, first_stride, basic_block=basic_block),
            Efficient_ViG_blocks(output_features, img_shape_list[d+1], d-conv_layer_d_num+1, conv_layer_d_num, opt=self.opt, conv_op=self.conv_op,
                                norm_op=self.norm_op, norm_op_kwargs=self.norm_op_kwargs, dropout_op=self.dropout_op)))
        
        d_output_channels_list.append(final_num_features)

        # if we don't want to do dropout in the localization pathway then we set the dropout prob to zero here
        if not dropout_in_localization:
            old_dropout_p = self.dropout_op_kwargs['p']
            self.dropout_op_kwargs['p'] = 0.0

        u_output_channels_list = []
        # now lets build the localization pathway
        for u in range(num_pool):
            nfeatures_from_down = final_num_features
            nfeatures_from_skip = d_output_channels_list[-(2 + u)]
            n_features_after_tu_and_concat = nfeatures_from_skip * 2

            # the first conv reduces the number of features to match those of skip
            # the following convs work on that number of features
            # if not convolutional upsampling then the final conv reduces the num of features again
            if u != num_pool - 1 and not self.convolutional_upsampling:
                #final_num_features = self.conv_blocks_context[-(3 + u)].output_channels
                final_num_features = d_output_channels_list[-(3 + u)]
            else:
                final_num_features = nfeatures_from_skip

            if not self.convolutional_upsampling:
                self.tu.append(Upsample(scale_factor=pool_op_kernel_sizes[-(u + 1)], mode=upsample_mode))
            else:
                self.tu.append(transpconv(nfeatures_from_down, nfeatures_from_skip, pool_op_kernel_sizes[-(u + 1)],
                                          pool_op_kernel_sizes[-(u + 1)], bias=False))

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[- (u + 1)]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[- (u + 1)]

            if u < (num_pool-conv_layer_d_num):
                self.conv_blocks_localization.append(nn.Sequential(
                    StackedConvLayers(n_features_after_tu_and_concat, nfeatures_from_skip, num_conv_per_stage - 1,
                                    self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs, self.dropout_op,
                                    self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs, basic_block=basic_block),
                    Efficient_ViG_blocks(nfeatures_from_skip, img_shape_list[num_pool - (u+1)], num_pool-conv_layer_d_num - (u+1), conv_layer_d_num, opt=self.opt, conv_op=self.conv_op,
                                    norm_op=self.norm_op, norm_op_kwargs=self.norm_op_kwargs, dropout_op=self.dropout_op)))
            else:
                self.conv_blocks_localization.append(nn.Sequential(
                    StackedConvLayers(n_features_after_tu_and_concat, nfeatures_from_skip, num_conv_per_stage - 1,
                                    self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs, self.dropout_op,
                                    self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs, basic_block=basic_block),
                    StackedConvLayers(nfeatures_from_skip, final_num_features, 1, self.conv_op, self.conv_kwargs,
                                    self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                                    self.nonlin, self.nonlin_kwargs, basic_block=basic_block)))
            
            u_output_channels_list.append(final_num_features)


        for ds in range(len(self.conv_blocks_localization)):
            output_channels_features = u_output_channels_list[ds]
            self.seg_outputs.append(conv_op(output_channels_features, num_classes,
                                            1, 1, 0, 1, 1, seg_output_use_bias))

        self.upscale_logits_ops = []
        cum_upsample = np.cumprod(np.vstack(pool_op_kernel_sizes), axis=0)[::-1]
        for usl in range(num_pool - 1):
            if self.upscale_logits:
                self.upscale_logits_ops.append(Upsample(scale_factor=tuple([int(i) for i in cum_upsample[usl + 1]]),
                                                        mode=upsample_mode))
            else:
                self.upscale_logits_ops.append(lambda x: x)

        if not dropout_in_localization:
            self.dropout_op_kwargs['p'] = old_dropout_p

        # register all modules properly
        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)
        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)
        self.td = nn.ModuleList(self.td)
        self.tu = nn.ModuleList(self.tu)
        self.seg_outputs = nn.ModuleList(self.seg_outputs)
        if self.upscale_logits:
            self.upscale_logits_ops = nn.ModuleList(
                self.upscale_logits_ops)  # lambda x:x is not a Module so we need to distinguish here

        if self.weightInitializer is not None:
            self.apply(self.weightInitializer)

    def forward(self, x):
        skips = []
        seg_outputs = []
        for d in range(len(self.conv_blocks_context) - 1):
            # pos_embed:
            if d == self.conv_layer_d_num:
                x = x + self.pos_embed.clone() # https://github.com/NVlabs/FUNIT/issues/23
            else:
                pass

            x = self.conv_blocks_context[d](x)
            skips.append(x)
            if not self.convolutional_pooling:
                x = self.td[d](x)

        x = self.conv_blocks_context[-1](x)

        for u in range(len(self.tu)):
            x = self.tu[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

        if self._deep_supervision and self.do_ds:
            return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            return seg_outputs[-1]

    @staticmethod
    def compute_approx_vram_consumption(patch_size, num_pool_per_axis, base_num_features, max_num_features,
                                        num_modalities, num_classes, pool_op_kernel_sizes, deep_supervision=False,
                                        conv_per_stage=2):
        """
        This only applies for num_conv_per_stage and convolutional_upsampling=True
        not real vram consumption. just a constant term to which the vram consumption will be approx proportional
        (+ offset for parameter storage)
        :param deep_supervision:
        :param patch_size:
        :param num_pool_per_axis:
        :param base_num_features:
        :param max_num_features:
        :param num_modalities:
        :param num_classes:
        :param pool_op_kernel_sizes:
        :return:
        """
        if not isinstance(num_pool_per_axis, np.ndarray):
            num_pool_per_axis = np.array(num_pool_per_axis)

        npool = len(pool_op_kernel_sizes)

        map_size = np.array(patch_size)
        tmp = np.int64((conv_per_stage * 2 + 1) * np.prod(map_size, dtype=np.int64) * base_num_features +
                       num_modalities * np.prod(map_size, dtype=np.int64) +
                       num_classes * np.prod(map_size, dtype=np.int64))

        num_feat = base_num_features

        for p in range(npool):
            for pi in range(len(num_pool_per_axis)):
                map_size[pi] /= pool_op_kernel_sizes[p][pi]
            num_feat = min(num_feat * 2, max_num_features)
            num_blocks = (conv_per_stage * 2 + 1) if p < (npool - 1) else conv_per_stage  # conv_per_stage + conv_per_stage for the convs of encode/decode and 1 for transposed conv
            tmp += num_blocks * np.prod(map_size, dtype=np.int64) * num_feat
            if deep_supervision and p < (npool - 2):
                tmp += np.prod(map_size, dtype=np.int64) * num_classes
            # print(p, map_size, num_feat, tmp)
        return tmp
