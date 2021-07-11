#!/bin/sh
gpu_id=2
cd ..

mkdir baseline_cluster

for model in  AUTOGCN ChebNet GatedGCN GraphSage MLP GCN GIN GAT MoNet; do
    python main_SBMs_node_classification.py --gpu_id $gpu_id --config './configs/SBMs_node_clustering_'$model'_CLUSTER.json' --runs 5 > ./baseline_cluster/$model.log
done

