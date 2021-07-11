import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import numpy as np
from torch.nn import Linear, ReLU, Dropout

"""
Paper: https://arxiv.org/abs/1810.05997
Author's code: https://github.com/klicperajo/ppnp
"""
from dgl.nn.pytorch.conv import SGConv


class SGCNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()

        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        dropout = net_params['dropout']
        self.graph_norm = net_params['graph_norm']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.n_classes = n_classes
        self.device = net_params['device']

        self.embedding_h = nn.Linear(in_dim, hidden_dim)

        n_layers = net_params['L']
        layers = []
        for i in range(n_layers-1):
            layers.append(Linear(hidden_dim,hidden_dim))
        self.layers = nn.ModuleList(layers)
        self.act_fn = ReLU()
        self.dropout = Dropout(p=dropout)
        self.prop = SGConv(hidden_dim,
                   n_classes,
                   k=2,
                   cached=False,
                   bias=True)

    def forward(self, g, h, e, snorm_n, snorm_e):
        h = self.embedding_h(h)
        for i, layer in enumerate(self.layers):
            h = layer(self.dropout(h))
            if i == len(self.layers) - 1:
                break
            h = self.act_fn(h)

        h = self.prop(g, h)
        return h

    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)

        return loss











