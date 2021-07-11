#!/bin/sh
gpu_id=0
cd ..

mkdir baseline_mnist

for model in  AUTOGCN ChebNet GatedGCN GraphSage MLP GCN GIN GAT MoNet; do
    python main_superpixels_graph_classification.py --config './configs/superpixels_graph_classification_'$model'_MNIST.json' --runs 5 --gpu_id $gpu_id  > ./baseline_mnist/$model.log
done