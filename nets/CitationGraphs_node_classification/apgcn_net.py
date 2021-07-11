import math
import torch
from typing import List
from torch.nn import ModuleList, Dropout, ReLU, Linear
from torch import nn
import dgl.function as fn
from layers.ap_layer import APLayer

class APGCNNet(torch.nn.Module):
    def __init__(self, net_params):
        super(APGCNNet, self).__init__()

        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        n_layers = net_params['L']
        n_classes = net_params['n_classes']
        n_iter = net_params['n_iter']
        prop_penalty = net_params['prop_penalty']
        dropout = net_params['dropout']

        layers = []
        layers.append(Linear(in_dim, hidden_dim))
        for i in range(n_layers-1):
            layers.append(Linear(hidden_dim,hidden_dim))
        layers.append(Linear(hidden_dim,n_classes))
        self.prop = APLayer(n_iter, n_classes)
        self.prop_penalty = prop_penalty
        self.layers = ModuleList(layers)
        self.reg_params = list(layers[0].parameters())
        self.non_reg_params = list([p for l in layers[1:] for p in l.parameters()])
        self.dropout = Dropout(p=dropout)
        self.act_fn = ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        self.prop.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, g, h, e, snorm_n, snorm_e):

        for i, layer in enumerate(self.layers):
            h = layer(self.dropout(h))
            if i == len(self.layers) - 1:
                break
            h = self.act_fn(h)
        h, steps, reminders = self.prop(g,h)
        return h, steps, reminders

    def loss(self, pred, label, steps, reminders):

        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label) + self.prop_penalty * (steps + reminders).mean()

        return loss

