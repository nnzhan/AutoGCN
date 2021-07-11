import torch.nn as nn
from layers.cheb_layer import ChebLayer
from layers.mlp_readout_layer import MLPReadout
import dgl
import torch.nn.functional as F
import torch
import math
import torch
from typing import List
from torch.nn import ModuleList, Dropout, ReLU, Linear
from torch import nn
import dgl.function as fn
from layers.ap_layer import APLayer

class APGCNNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        in_dim_node = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        n_classes = net_params['n_classes']
        self.n_classes = n_classes
        n_layers = net_params['L']
        n_iter = net_params['n_iter']
        prop_penalty = net_params['prop_penalty']
        dropout = net_params['dropout']

        self.embedding_h = nn.Embedding(in_dim_node, hidden_dim)
        self.in_feat_dropout = nn.Dropout(dropout)

        layers = []
        for i in range(n_layers-1):
            layers.append(Linear(hidden_dim,hidden_dim))
        self.prop = APLayer(n_iter, hidden_dim)
        self.prop_penalty = prop_penalty
        self.layers = ModuleList(layers)
        self.reg_params = list(layers[0].parameters())
        self.non_reg_params = list([p for l in layers[1:] for p in l.parameters()])
        self.dropout = Dropout(p=dropout)
        self.act_fn = ReLU()
        self.reset_parameters()
        self.device = net_params['device']
        self.MLP_layer = MLPReadout(hidden_dim, n_classes)


    def reset_parameters(self):
        self.prop.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()


    def forward(self, g, h, e, snorm_n, snorm_e):
        # input embedding
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)

        for i, layer in enumerate(self.layers):
            h = layer(self.dropout(h))
            if i == len(self.layers) - 1:
                break
            h = self.act_fn(h)
        h, steps, reminders = self.prop(g,h)
        h = self.MLP_layer(h)
        return h, steps, reminders

    def loss(self, pred, label, steps, reminders):
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
        loss = criterion(pred, label) + self.prop_penalty * (steps + reminders).mean()
        return loss




