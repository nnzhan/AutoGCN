#!/bin/sh
gpu_id=0
cd ..

mkdir log_layer
for L in 2 4 6 8 10 12 14 16; do
    python main_molecules_graph_regression.py --gpu_id $gpu_id --config './configs/molecules_graph_regression_GCN_ZINC.json' --L $L --runs 5 > ./log_layer/GCN-molecule-L$L.log
done