import torch.nn as nn
import torch.nn.functional as F
from layers.autogcn_layer import AUTOGCNLayer

class AUTOGCNNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        n_classes = net_params['n_classes']
        graph_norm = net_params['graph_norm']
        batch_norm = net_params['batch_norm']
        residual = net_params['residual']
        dropout = net_params['dropout']
        K = net_params['K']
        n_layers = net_params['L']

        num_filters = net_params['num_filters']
        opt = net_params['opt']
        gate = net_params['gate']

        self.dropout = dropout
        self.device = net_params['device']
        self.layers = nn.ModuleList()
        #self.layers.append(nn.Linear(in_dim,hidden_dim))
        self.layers.append(AUTOGCNLayer(in_dim, hidden_dim, F.relu, dropout, graph_norm, batch_norm, num_filters=num_filters, K=K, residual=residual, gate=gate, opt=opt))
        self.layers.extend(nn.ModuleList([AUTOGCNLayer(hidden_dim, hidden_dim, F.relu, dropout, graph_norm, batch_norm, num_filters=num_filters, K=K, residual=residual, gate=gate, opt=opt) for _ in range(n_layers - 1)]))
        self.layers.append(AUTOGCNLayer(hidden_dim, n_classes, None, dropout, graph_norm, batch_norm, num_filters=num_filters, K=K, residual=residual, gate=gate, opt=opt))


    def forward(self, g, h, e, snorm_n, snorm_e):
        # h = self.layers[0](h)
        # h = self.layers[1](g, h, snorm_n)
        for conv in self.layers:
            h = conv(g, h, snorm_n)
        return h

    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss


