import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import numpy as np

"""
Paper: https://arxiv.org/abs/1810.05997
Author's code: https://github.com/klicperajo/ppnp
"""
from dgl.nn.pytorch.conv import SGConv
from layers.mlp_readout_layer import MLPReadout
from torch.nn import Linear, ReLU

class SGCNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()

        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        n_classes = net_params['n_classes']
        dropout = net_params['dropout']
        self.n_classes = n_classes
        self.device = net_params['device']
        self.embedding_h = nn.Embedding(in_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        n_layers = net_params['L']
        layers = []
        for i in range(n_layers-1):
            layers.append(Linear(hidden_dim,hidden_dim))
        self.layers = nn.ModuleList(layers)
        self.act_fn = ReLU()
        self.prop = SGConv(hidden_dim,
                           hidden_dim,
                           k=2,
                           cached=False,
                           bias=True)
        self.MLP_layer = MLPReadout(hidden_dim, n_classes)

    def forward(self, g, h, e, snorm_n, snorm_e):
        h = self.embedding_h(h)
        h = self.dropout(h)
        for i, layer in enumerate(self.layers):
            h = layer(self.dropout(h))
            if i == len(self.layers) - 1:
                break
            h = self.act_fn(h)
        h = self.prop(g, h)
        h = self.MLP_layer(h)
        return h

    def loss(self, pred, label):
        # calculating label weights for weighted loss computation
        V = label.size(0)
        label_count = torch.bincount(label)
        label_count = label_count[label_count.nonzero()].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes > 0).float()

        # weighted cross-entropy for unbalanced classes
        criterion = nn.CrossEntropyLoss(weight=weight)
        loss = criterion(pred, label)

        return loss


