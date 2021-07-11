#!/bin/sh
gpu_id=0
cd ..

mkdir log_layer

for L in 2 4 6 8 10 12 14 16; do
    python main_SBMs_node_classification.py --gpu_id $gpu_id --config './configs/SBMs_node_clustering_AUTOGCN_CLUSTER.json' --L $L --runs 5 > ./log_layer/AUTOGCN-cluster-L$L.log
done

