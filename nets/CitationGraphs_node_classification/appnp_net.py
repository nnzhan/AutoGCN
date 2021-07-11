import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import numpy as np

"""
Paper: https://arxiv.org/abs/1810.05997
Author's code: https://github.com/klicperajo/ppnp
"""
from layers.appnp_layer import APPNP

class APPNPNet(nn.Module):

    def __init__(self, net_params):
        super().__init__()

        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.graph_norm = net_params['graph_norm']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.n_classes = n_classes
        self.device = net_params['device']
        self.dgl_builtin = net_params['builtin']

        self.layer = APPNP(in_dim,[hidden_dim for i in range(n_layers)],n_classes,F.relu,dropout,0.1,10)


    def forward(self, g, h, e, snorm_n, snorm_e):
        h = self.layer(g,h)
        return h

    
    def loss(self, pred, label):

        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)

        return loss











