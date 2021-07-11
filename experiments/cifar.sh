#!/bin/sh
gpu_id=0
cd ..

mkdir baseline_cifar
for model in  AUTOGCN ChebNet GatedGCN GraphSage MLP GCN GIN GAT MoNet; do
    python main_superpixels_graph_classification.py --gpu_id $gpu_id --config './configs/superpixels_graph_classification_'$model'_CIFAR10.json' --runs 5 > ./baseline_cifar/$model.log
done