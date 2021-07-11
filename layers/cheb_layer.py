import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn
from dgl.nn.pytorch import ChebConv
from torch.nn import init


# Sends a message of node feature h
# Equivalent to => return {'m': edges.src['h']}

class MPConv(nn.Module):
    def __init__(self):
        super(MPConv, self).__init__()

    def forward(self, graph, feat):
        graph = graph.local_var()
        norm = torch.pow(graph.in_degrees().float().clamp(min=1), -0.5)
        shp = norm.shape + (1,) * (feat.dim() - 1)
        norm = torch.reshape(norm, shp).to(feat.device)

        # normalization by src node
        feat = feat * norm
        graph.ndata['h'] = feat
        graph.update_all(fn.copy_src(src='h', out='m'),
                         fn.sum(msg='m', out='h'))

        feat = graph.ndata['h']
        # normalization by dst node
        feat = feat * norm
        return feat

class ChebLayer(nn.Module):
    """
    Second order approximation, with K=2, lambda_max=2
        Param: [in_dim, out_dim]
    """
    def __init__(self, in_dim, out_dim, activation, dropout, graph_norm, batch_norm, residual=False):
        super().__init__()
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.residual = residual

        if in_dim != out_dim:
            self.residual = False
        
        self.batchnorm_h = nn.BatchNorm1d(out_dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        self.mp = MPConv()
        self.c1 = nn.Linear(in_dim, out_dim,bias=False)
        self.c2 = nn.Linear(in_dim, out_dim,bias=False)
        self.c3 = nn.Linear(in_dim, out_dim,bias=False)

        self.bias = nn.Parameter(torch.Tensor(out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            init.zeros_(self.bias)

        
    def forward(self, g, feature, snorm_n):

        h_in = feature   # to be used for residual connection
        ax = self.mp(g, feature)
        ax2 = self.mp(g, ax)
        hc1 = self.c1(feature)
        hc2 = self.c2(-ax)
        hc3 = self.c3(2*ax2-feature)
        h = hc1 + hc2 + hc3 + self.bias

        if self.graph_norm:
            h = h * snorm_n # normalize activation w.r.t. graph size

        if self.batch_norm:
            h = self.batchnorm_h(h) # batch normalization  
       
        if self.activation:
            h = self.activation(h)
        
        if self.residual:
            h = h_in + h # residual connection

        h = self.dropout(h)
        return h
    
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.residual)
