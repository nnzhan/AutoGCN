import torch.nn as nn
from layers.cheb_layer import ChebLayer
from layers.mlp_readout_layer import MLPReadout
import dgl
import torch.nn.functional as F

class ChebNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        graph_norm = net_params['graph_norm']
        batch_norm = net_params['batch_norm']
        residual = net_params['residual']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        in_feat_dropout = net_params['in_feat_dropout']
        n_classes = net_params['n_classes']

        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        self.readout = net_params['readout']
        self.device = net_params['device']
        self.layers = nn.ModuleList()
        self.layers = nn.ModuleList([ChebLayer(hidden_dim, hidden_dim,
            F.relu, dropout, graph_norm, batch_norm, residual=residual) for _ in range(n_layers)])
        self.MLP_layer = MLPReadout(hidden_dim, n_classes)   # 1 out dim since regression problem



    def forward(self, g, h, e, snorm_n, snorm_e):
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)

        for conv in self.layers:
            h = conv(g, h, snorm_n)
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

    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss


