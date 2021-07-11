"""
    Utility file to select GraphNN model as
    selected by the user
"""

from nets.CitationGraphs_node_classification.gcn_net import GCNNet
from nets.CitationGraphs_node_classification.gat_net import GATNet
from nets.CitationGraphs_node_classification.graphsage_net import GraphSageNet
from nets.CitationGraphs_node_classification.mlp_net import MLPNet
from nets.CitationGraphs_node_classification.autogcn_net import AUTOGCNNet
from nets.CitationGraphs_node_classification.mo_net import MoNet as MoNet_
from nets.CitationGraphs_node_classification.gated_gcn_net import GatedGCNNet
from nets.CitationGraphs_node_classification.gin_net import GINNet
from nets.CitationGraphs_node_classification.appnp_net import APPNPNet
from nets.CitationGraphs_node_classification.sgc_net import SGCNet
from nets.CitationGraphs_node_classification.cheb_net import ChebNet

from nets.CitationGraphs_node_classification.apgcn_net import APGCNNet

def GCN(net_params):
    return GCNNet(net_params)

def GAT(net_params):
    return GATNet(net_params)

def GraphSage(net_params):
    return GraphSageNet(net_params)

def MLP(net_params):
    return MLPNet(net_params)

def AUTOGCN(net_params):
    return AUTOGCNNet(net_params)

def MoNet(net_params):
    return MoNet_(net_params)

def GatedGCN(net_params):
    return GatedGCNNet(net_params)

def GIN(net_params):
    return GINNet(net_params)

def APPNP(net_params):
    return APPNPNet(net_params)

def SGC(net_params):
    return SGCNet(net_params)

def CHEB(net_params):
    return ChebNet(net_params)

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
        'MLP': MLP,
        'AUTOGCN': AUTOGCN,
        'APPNP': APPNP,
        'SGC': SGC,
        'ChebNet': CHEB,
        'APGCN': APGCN
    }
    return models[MODEL_NAME](net_params)
