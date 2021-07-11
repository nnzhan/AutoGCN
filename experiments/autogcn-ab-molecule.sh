#!/bin/sh
gpu_id=0
cd ..

mkdir log_ablation

#w/o over
python main_molecules_graph_regression.py --gpu_id $gpu_id --config './configs/molecules_graph_regression_AUTOGCN_ZINC.json' --L 8 --runs 5 --opt single > ./log_ablation/AUTOGCN-molecule-single.log

#w/o par
python main_molecules_graph_regression.py --gpu_id $gpu_id --config './configs/molecules_graph_regression_AUTOGCN_ZINC.json' --L 8 --runs 5 --opt fix > ./log_ablation/AUTOGCN-molecule-fix.log

#w/o gate
python main_molecules_graph_regression.py --gpu_id $gpu_id --config './configs/molecules_graph_regression_AUTOGCN_ZINC.json' --L 8 --runs 5 --gate False > ./log_ablation/AUTOGCN-molecule-wogate.log





