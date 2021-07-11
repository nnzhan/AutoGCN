import torch
from torch.nn import ModuleList, Dropout, ReLU, Linear
from torch import nn
from layers.ap_layer import APLayer
import dgl

"""
    GCN: Graph Convolutional Networks
    Thomas N. Kipf, Max Welling, Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)
    http://arxiv.org/abs/1609.02907
"""
from layers.mlp_readout_layer import MLPReadout


class APGCNNet(torch.nn.Module):
    def __init__(self, net_params):
        super(APGCNNet, self).__init__()
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        n_classes = net_params['n_classes']
        n_layers = net_params['L']
        n_iter = net_params['n_iter']
        prop_penalty = net_params['prop_penalty']
        dropout = net_params['dropout']

        layers = []
        for i in range(n_layers-1):
            layers.append(Linear(hidden_dim,hidden_dim))
        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        self.prop = APLayer(n_iter, hidden_dim)
        self.prop_penalty = prop_penalty
        self.layers = ModuleList(layers)
        self.reg_params = list(layers[0].parameters())
        self.non_reg_params = list([p for l in layers[1:] for p in l.parameters()])
        self.dropout = Dropout(p=dropout)
        self.act_fn = ReLU()
        self.reset_parameters()
        self.MLP_layer = MLPReadout(hidden_dim, n_classes)   # 1 out dim since regression problem
        self.readout = net_params['readout']
        in_feat_dropout = net_params['in_feat_dropout']
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)


    def reset_parameters(self):
        self.prop.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, g, h, e, snorm_n, snorm_e):
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        for i, layer in enumerate(self.layers):
            h = layer(self.dropout(h))
            if i == len(self.layers) - 1:
                break
            h = self.act_fn(h)
        h, steps, reminders = self.prop(g,h)

        g.ndata['h'] = h


        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes
        return self.MLP_layer(hg), steps, reminders

    def loss(self, scores, targets, steps, reminders):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(scores, targets) + self.prop_penalty * (steps + reminders).mean()
        return loss
