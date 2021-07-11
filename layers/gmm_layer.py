import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import dgl.function as fn
from dgl.nn.pytorch.conv import GMMConv

"""
    GMM: Gaussian Mixture Model Convolution layer
    Geometric Deep Learning on Graphs and Manifolds using Mixture Model CNNs (Federico Monti et al., CVPR 2017)
    https://arxiv.org/pdf/1611.08402.pdf
"""

class GMMLayer(nn.Module):
    """
    [!] code adapted from dgl implementation of GMMConv

    Parameters
    ----------
    in_dim : 
        Number of input features.
    out_dim : 
        Number of output features.
    dim : 
        Dimensionality of pseudo-coordinte.
    kernel : 
        Number of kernels :math:`K`.
    aggr_type : 
        Aggregator type (``sum``, ``mean``, ``max``).
    dropout :
        Required for dropout of output features.
    graph_norm : 
        boolean flag for output features normalization w.r.t. graph sizes.
    batch_norm :
        boolean flag for batch_norm layer.
    residual : 
        If True, use residual connection inside this layer. Default: ``False``.
    bias : 
        If True, adds a learnable bias to the output. Default: ``True``.
    
    """
    def __init__(self, in_dim, out_dim, activation, dim, kernel, aggr_type, dropout,
                 graph_norm, batch_norm, residual=False, bias=True):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dim = dim
        self.kernel = kernel
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.residual = residual
        self.dropout = dropout
        self.activation = activation

        self.layer = GMMConv(in_dim, out_dim, dim, kernel,aggr_type)

        self.bn_node_h = nn.BatchNorm1d(out_dim)
        if in_dim != out_dim:
            self.residual = False
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_dim))
        else:
            self.register_buffer('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        if self.bias is not None:
            init.zeros_(self.bias.data)
    
    def forward(self, g, h, pseudo, snorm_n):
        h_in = h # for residual connection
        
        h = self.layer(g, h, pseudo)
        
        if self.graph_norm:
            h = h* snorm_n # normalize activation w.r.t. graph size
        
        if self.batch_norm:
            h = self.bn_node_h(h) # batch normalization  
        
        if self.activation:
            h = self.activation(h)

        if self.residual:
            h = h_in + h # residual connection
        
        if self.bias is not None:
            h = h + self.bias
        
        h = F.dropout(h, self.dropout, training=self.training)
        
        return h
