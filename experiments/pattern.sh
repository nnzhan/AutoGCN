#!/bin/sh
gpu_id=0
cd ..

mkdir baseline_pattern
for model in  AUTOGCN ChebNet GatedGCN GraphSage MLP GCN GIN GAT MoNet; do
    python main_SBMs_node_classification.py --gpu_id $gpu_id --config './configs/SBMs_node_clustering_'$model'_PATTERN.json' --runs 5 > ./baseline_pattern/$model.log
done

