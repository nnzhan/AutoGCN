"""
    Utility file to select GraphNN model as
    selected by the user
"""

from nets.superpixels_graph_classification.gated_gcn_net import GatedGCNNet
from nets.superpixels_graph_classification.gcn_net import GCNNet
from nets.superpixels_graph_classification.gat_net import GATNet
from nets.superpixels_graph_classification.graphsage_net import GraphSageNet
from nets.superpixels_graph_classification.gin_net import GINNet
from nets.superpixels_graph_classification.mo_net import MoNet as MoNet_
from nets.superpixels_graph_classification.diffpool_net import DiffPoolNet
from nets.superpixels_graph_classification.mlp_net import MLPNet
from nets.superpixels_graph_classification.autogcn_net import AUTOGCNNet
from nets.superpixels_graph_classification.cheb_net import ChebNet
from nets.superpixels_graph_classification.sgc_net import SGCNet
from nets.superpixels_graph_classification.apgcn_net import APGCNNet

def GatedGCN(net_params):
    return GatedGCNNet(net_params)

def GCN(net_params):
    return GCNNet(net_params)

def GAT(net_params):
    return GATNet(net_params)

def GraphSage(net_params):
    return GraphSageNet(net_params)

def GIN(net_params):
    return GINNet(net_params)

def MoNet(net_params):
    return MoNet_(net_params)

def DiffPool(net_params):
    return DiffPoolNet(net_params)

def MLP(net_params):
    return MLPNet(net_params)

def AUTOGCN(net_params):
    return AUTOGCNNet(net_params)

def CHEB(net_params):
    return ChebNet(net_params)

def SGC(net_params):
    return SGCNet(net_params)

def APGCN(net_params):
    return APGCNNet(net_params)

def gnn_model(MODEL_NAME, net_params):
    models = {
        'GatedGCN': GatedGCN,
        'GCN': GCN,
        'GAT': GAT,
        'GraphSage': GraphSage,
        'GIN': GIN,
        'MoNet': MoNet,
        'DiffPool': DiffPool,
        'MLP': MLP,
        'AUTOGCN': AUTOGCN,
        'ChebNet': CHEB,
        'SGC': SGC,
        "APGCN": APGCN
    }
        
    return models[MODEL_NAME](net_params)