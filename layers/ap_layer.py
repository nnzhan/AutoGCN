import math
import torch
from typing import List
from torch.nn import ModuleList, Dropout, ReLU, Linear
from torch import nn
import dgl.function as fn

class APLayer(nn.Module):
    def __init__(self, niter, h_size):
        super(APLayer, self).__init__()

        self.niter = niter
        self.halt = Linear(h_size, 1)
        self.reg_params = list(self.halt.parameters())
        self.dropout = Dropout()
        self.reset_parameters()
        self.edge_drop = nn.Dropout(0.0)

    def reset_parameters(self):
        self.halt.reset_parameters()
        x = (self.niter + 1) // 1
        b = math.log((1 / x) / (1 - (1 / x)))
        self.halt.bias.data.fill_(b)

    def forward(self, g, feature):
        sz = feature.size(0)
        steps = torch.ones(sz).to(feature.device)
        sum_h = torch.zeros(sz).to(feature.device)
        continue_mask = torch.ones(sz, dtype=torch.bool).to(feature.device)
        x = torch.zeros_like(feature).to(feature.device)

        prop = self.dropout(feature)
        for _ in range(self.niter):

            old_prop = prop
            continue_fmask = continue_mask.type('torch.FloatTensor').to(feature.device)
            prop = self.propagate(g, feature)
            h = torch.sigmoid(self.halt(prop)).t().squeeze()

            prob_mask = (((sum_h + h) < 0.99) & continue_mask).squeeze()
            prob_fmask = prob_mask.type('torch.FloatTensor').to(feature.device)

            steps = steps + prob_fmask
            sum_h = sum_h + prob_fmask * h

            final_iter = steps < self.niter

            condition = prob_mask & final_iter
            p = torch.where(condition, sum_h, 1 - sum_h)

            to_update = self.dropout(continue_fmask)[:, None]
            x = x + (prop * p[:, None] +
                     old_prop * (1 - p)[:, None]) * to_update

            continue_mask = continue_mask & prob_mask

            if (~continue_mask).all():
                break
        x = x / steps[:, None]
        return x, steps, (1 - sum_h)

    def propagate(self, g, feat):
        with g.local_scope():
            src_norm = torch.pow(g.out_degrees().float().clamp(min=1), -0.5)
            shp = src_norm.shape + (1,) * (feat.dim() - 1)
            src_norm = torch.reshape(src_norm, shp).to(feat.device)
            dst_norm = torch.pow(g.in_degrees().float().clamp(min=1), -0.5)
            shp = dst_norm.shape + (1,) * (feat.dim() - 1)
            dst_norm = torch.reshape(dst_norm, shp).to(feat.device)

            feat = feat * src_norm
            g.ndata['h'] = feat
            g.edata['w'] = self.edge_drop(torch.ones(g.number_of_edges(), 1).to(feat.device))

            g.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum('m', 'h'))
            feat = g.ndata.pop('h')
            feat = feat * dst_norm
        return feat

