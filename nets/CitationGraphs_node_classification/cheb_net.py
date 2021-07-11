import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import numpy as np


from layers.cheb_layer import ChebLayer

class ChebNet(nn.Module):

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

        self.layers = nn.ModuleList()
        # input
        self.layers.append(ChebLayer(in_dim, hidden_dim, F.relu, dropout,
            self.graph_norm, self.batch_norm, residual=self.residual))

        # hidden
        self.layers.extend(nn.ModuleList([ChebLayer(hidden_dim, hidden_dim,
            F.relu, dropout, self.graph_norm, self.batch_norm, residual=self.residual)
            for _ in range(n_layers-1)]))

        # output
        self.layers.append(ChebLayer(hidden_dim, n_classes, None, 0,
            self.graph_norm, self.batch_norm, residual=self.residual))

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, h, e, snorm_n, snorm_e):
      
        # GCN
        for i, conv in enumerate(self.layers):
            h = conv(g, h, snorm_n)
        return h

    
    def loss(self, pred, label):

        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)

        return loss











