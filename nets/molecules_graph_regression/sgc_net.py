import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from torch.nn import Linear, ReLU, Dropout

"""
    GCN: Graph Convolutional Networks
    Thomas N. Kipf, Max Welling, Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)
    http://arxiv.org/abs/1609.02907
"""
from layers.gcn_layer import GCNLayer
from layers.mlp_readout_layer import MLPReadout
from dgl.nn.pytorch.conv import SGConv


class SGCNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        num_atom_type = net_params['num_atom_type']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']

        self.readout = net_params['readout']
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.embedding_h = nn.Embedding(num_atom_type, hidden_dim)
        n_layers = net_params['L']
        layers = []
        for i in range(n_layers-1):
            layers.append(Linear(hidden_dim,hidden_dim))
        self.layers = nn.ModuleList(layers)
        self.act_fn = ReLU()
        self.dropout = Dropout(p=dropout)
        self.prop = SGConv(hidden_dim,
                           hidden_dim,
                           k=8,
                           cached=False,
                           bias=True)
        self.MLP_layer = MLPReadout(out_dim, 1)  # 1 out dim since regression problem

    def forward(self, g, h, e, snorm_n, snorm_e):
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)

        for i, layer in enumerate(self.layers):
            h = layer(self.dropout(h))
            if i == len(self.layers) - 1:
                break
            h = self.act_fn(h)

        h = self.prop(g, h)

        g.ndata['h'] = h

        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes

        return self.MLP_layer(hg)

    def loss(self, scores, targets):
        # loss = nn.MSELoss()(scores,targets)
        loss = nn.L1Loss()(scores, targets)
        return loss
