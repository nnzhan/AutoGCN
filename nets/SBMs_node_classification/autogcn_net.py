import torch.nn as nn
from layers.autogcn_layer import AUTOGCNLayer

from dgl.nn.pytorch.conv import SGConv

from layers.mlp_readout_layer import MLPReadout
import dgl
import torch.nn.functional as F
import torch


class AUTOGCNNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        in_dim_node = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        graph_norm = net_params['graph_norm']
        batch_norm = net_params['batch_norm']
        residual = net_params['residual']
        dropout = net_params['dropout']
        K = net_params['K']
        n_layers = net_params['L']
        in_feat_dropout = net_params['in_feat_dropout']
        num_filters = net_params['num_filters']
        opt = net_params['opt']
        gate = net_params['gate']

        self.n_classes = n_classes
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.embedding_h = nn.Embedding(in_dim_node, hidden_dim)

        self.readout = net_params['readout']
        self.device = net_params['device']
        self.layers = nn.ModuleList()
        self.layers = nn.ModuleList([AUTOGCNLayer(hidden_dim, hidden_dim, F.relu, dropout, graph_norm, batch_norm, num_filters=num_filters, opt=opt, K=K, residual=residual, gate=gate) for _ in range(n_layers)])
        self.MLP_layer = MLPReadout(hidden_dim, n_classes)

    def forward(self, g, h, e, snorm_n, snorm_e):
        # input embedding
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        # GCN
        for conv in self.layers:
            h = conv(g, h, snorm_n)
        # output
        h_out = self.MLP_layer(h)

        return h_out

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




