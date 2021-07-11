#!/bin/sh
gpu_id=0
cd ..

mkdir baseline_molecule

for model in  AUTOGCN ChebNet GatedGCN GraphSage MLP GCN GIN GAT MoNet; do
    python main_molecules_graph_regression.py --gpu_id $gpu_id --config './configs/molecules_graph_regression_'$model'_ZINC.json' --runs 5 > ./baseline_molecule/$model.log
done



