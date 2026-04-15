import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torchvision.transforms as T
from torchvision.utils import save_image
from typing import Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn as nn
import os
import math
import argparse
import pprint
import copy
import torch_geometric as tg
from torch_geometric.nn import GATConv
from torch.autograd import Variable
class ElementWiseMulLayer(nn.Module):
    def __init__(self):
        super(ElementWiseMulLayer, self).__init__()
    
    def forward(self, u, logp,m):
        # 直接使用*运算符进行element-wise乘法
        if u.dim() == 4:
            u = u[:,:,:,0]
        return u * logp+m
def create_masks(input_size, hidden_size, n_hidden, input_order='sequential', input_degrees=None):
    # MADE paper sec 4:
    # degrees of connections between layers -- ensure at most in_degree - 1 connections
    degrees = []

    # set input degrees to what is provided in args (the flipped order of the previous layer in a stack of mades);
    # else init input degrees based on strategy in input_order (sequential or random)
    if input_order == 'sequential':
        degrees += [torch.arange(input_size)] if input_degrees is None else [input_degrees]
        for _ in range(n_hidden + 1):
            degrees += [torch.arange(hidden_size) % (input_size - 1)]
        degrees += [torch.arange(input_size) % input_size - 1] if input_degrees is None else [input_degrees % input_size - 1]

    elif input_order == 'random':
        degrees += [torch.randperm(input_size)] if input_degrees is None else [input_degrees]
        for _ in range(n_hidden + 1):
            min_prev_degree = min(degrees[-1].min().item(), input_size - 1)
            degrees += [torch.randint(min_prev_degree, input_size, (hidden_size,))]
        min_prev_degree = min(degrees[-1].min().item(), input_size - 1)
        degrees += [torch.randint(min_prev_degree, input_size, (input_size,)) - 1] if input_degrees is None else [input_degrees - 1]

    # construct masks
    masks = []
    for (d0, d1) in zip(degrees[:-1], degrees[1:]):
        masks += [(d1.unsqueeze(-1) >= d0.unsqueeze(0)).float()]

    return masks, degrees[0]



class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphConv, self).__init__()
        self.graph_linear_self1 = nn.Linear(in_channels*4, in_channels*2)
        self.graph_linear_self2 = nn.Linear(in_channels*2, in_channels)
        # self.gru = nn.GRU(in_channels, senums, 5,batch_first=True,bidirectional=False)
        # self.graph_linear_edge = nn.Linear(in_channels, out_channels * num_edge_type)
        # self.num_edge_type = num_edge_type
        self.in_ch = in_channels
        self.out_ch = out_channels
        # self.num_atoms = num_atoms

    def forward(self, x,adj) -> torch.Tensor:
        
        if x.dim() == 3:
            # xh=x.reshape(self.batch_size, self.timesteps*self.senums)
            hr = torch.einsum('ij,bkj->bkj', adj, x)
        else:
            x=x.reshape(x.shape[0], x.shape[1], x.shape[2]*4)
            hs = self.graph_linear_self1(x)
            hs = F.relu(hs)
            hs = self.graph_linear_self2(hs)
            hs = F.relu(hs)
            hr = torch.einsum('ij,bkj->bkj', adj, hs)



        return hr


class GraphMaskedLinear(nn.Linear):
    """ MADE building block layer """
    def __init__(self, input_size, n_outputs, mask):
        super().__init__(input_size, n_outputs)
        self.register_buffer('mask', mask)
        self.graph_conv = GraphConv(input_size, n_outputs)

    def forward(self, y,adj):

        h = self.graph_conv(y,adj)
        h = F.relu(h)
        out = F.linear(h, self.weight * self.mask, self.bias)

        return out

class MaskedLinear(nn.Linear):
    """ MADE building block layer """
    def __init__(self, input_size, n_outputs, mask, cond_label_size=None):
        super().__init__(input_size, n_outputs)

        self.register_buffer('mask', mask)

    def forward(self, x, y=None):
        out = F.linear(x, self.weight * self.mask, self.bias)

        return out




class FlowSequential(nn.Sequential):
    """ Container for layers of a normalizing flow """
    def forward(self,  y,x,adj):
        sum_log_abs_det_jacobians = 0
        for module in self:
            x, log_abs_det_jacobian = module( y,x,adj)
            sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + log_abs_det_jacobian
        return x, sum_log_abs_det_jacobians

    def inverse(self, u,adj):

        for module in reversed(self):
            u = module.inverse(u,adj)

        return u

# --------------------
# Models
# --------------------

class GMADE(nn.Module):
    def __init__(self, input_size, hidden_size, n_hidden, activation='relu', input_order='sequential', input_degrees=None, batch_size=64, timesteps=12, senums=207):
        """
        Args:
            input_size -- scalar; dim of inputs
            hidden_size -- scalar; dim of hidden layers
            n_hidden -- scalar; number of hidden layers
            activation -- str; activation function to use
            input_order -- str or tensor; variable order for creating the autoregressive masks (sequential|random)
                            or the order flipped from the previous layer in a stack of mades
            conditional -- bool; whether model is conditional
        """
        super().__init__()

        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(input_size))
        self.register_buffer('base_dist_var', torch.ones(input_size))

        # create masks
        masks, self.input_degrees = create_masks(input_size, hidden_size, n_hidden, input_order, input_degrees)


        # setup activation
        if activation == 'relu':
            activation_fn = nn.ReLU()
        elif activation == 'tanh':
            activation_fn = nn.Tanh()
        else:
            raise ValueError('Check activation function.')

        # construct model
        self.elayer = ElementWiseMulLayer()
        self.net_input = GraphMaskedLinear(input_size, hidden_size, masks[0])
        self.net = []
        for m in masks[1:-1]:
            self.net += [activation_fn, MaskedLinear(hidden_size, hidden_size, m)]
        self.net += [activation_fn, MaskedLinear(hidden_size,  2 * input_size, masks[-1].repeat(2,1))]
        self.net = nn.Sequential(*self.net)
        self.out_layer=nn.Linear(timesteps, 12)
        self.inverse_out_layer=nn.Linear(12,timesteps)
       


    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, y,x,adj):

        self.batch_size=y.shape[0]
        self.timesteps=y.shape[1]
        self.senums=y.shape[2]
        # MAF eq 4 -- return mean and log std
        h=self.net_input(y,adj)
        for layer in self.net:
            h = layer(h)
        
        h_hat=h.permute(0,2,1)
        out = self.out_layer(h_hat)
        h=out.permute(0,2,1)
        m, loga = h.chunk(chunks=2, dim=2)
        if x.dim() == 3:
            x_u = (x - m) * torch.exp(-loga)
        else:
            x_u = (x[:,:,:,0]- m) * torch.exp(-loga)
        # MAF eq 5
        log_abs_det_jacobian = - loga
        return x_u, log_abs_det_jacobian

    def inverse(self, u,adj):

        h = self.net_input(u, adj)
        for layer in self.net:
            h = layer(h)
        h_hat=h.permute(0,2,1)
        # out = self.out_layer(h_hat)   
        out= self.inverse_out_layer(h_hat)
        out=h_hat.permute(0,2,1)     
        m, loga = out.chunk(chunks=2, dim=2)
        y = self.elayer(u,torch.exp(loga),m)


        return y



    def log_prob(self, y,x,adj):
        u, log_abs_det_jacobian = self.forward(y,adj)
        return torch.sum(self.base_dist.log_prob(u) + log_abs_det_jacobian, dim=1)

class GMAF(nn.Module):
    def __init__(self, n_blocks, input_size, hidden_size, n_hidden, activation='relu', input_order='sequential', batch_size=64,timesteps=12, senums=207):
        super().__init__()
        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros((12,input_size)))
        self.register_buffer('base_dist_var', torch.zeros((12,input_size)))

        # construct model
        modules = []
        self.input_degrees = None
        for i in range(n_blocks):
            modules += [GMADE(input_size, hidden_size, n_hidden, activation, input_order, self.input_degrees,batch_size,timesteps, senums)]
            self.input_degrees = modules[-1].input_degrees.flip(0)
            # modules += batch_norm * [BatchNorm(input_size)]

        self.net = FlowSequential(*modules)
        self.inverse_out_layer=nn.Linear(12,timesteps)

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self,  y,x,adj):
        base_dist_mean = x[:,:,:,0].mean(dim=0)
        base_dist_var = x[:,:,:,0].var(dim=0)
        self.base_dist_mean=base_dist_mean*0.2+self.base_dist_mean*0.8
        self.base_dist_var=base_dist_var*0.2+self.base_dist_var*0.8
        return self.net( y,x,adj)

    def inverse(self, u, adj):
        y_hat=self.net.inverse(u, adj)
        y_hat=y_hat.permute(0,2,1)
        h=self.inverse_out_layer(y_hat)
        y_hat=h.permute(0,2,1)
        return y_hat

    def log_prob(self,  y,x,adj):
        u, sum_log_abs_det_jacobians = self.forward(y,x,adj)
        mean_u=u.mean(dim=0)
        return u,torch.sum(self.base_dist.log_prob(mean_u) + sum_log_abs_det_jacobians, dim=1)



