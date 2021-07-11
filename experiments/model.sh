#!/bin/sh
gpu_id=0
cd ..


model=AUTOGCN

python main_superpixels_graph_classification.py --config './configs/superpixels_graph_classification_'$model'_MNIST.json' --runs 5  --gpu_id $gpu_id > ./$model.log

python main_superpixels_graph_classification.py --config './configs/superpixels_graph_classification_'$model'_CIFAR10.json' --runs 5 --gpu_id $gpu_id > ./$model.log

python main_SBMs_node_classification.py --config './configs/SBMs_node_clustering_'$model'_CLUSTER.json' --runs 5 --gpu_id $gpu_id > ./$model.log

python main_SBMs_node_classification.py --config './configs/SBMs_node_clustering_'$model'_PATTERN.json' --runs 5 --gpu_id $gpu_id > ./$model.log

python main_molecules_graph_regression.py --config './configs/molecules_graph_regression_'$model'_ZINC.json' --runs 5 --gpu_id $gpu_id > ./$model.log
