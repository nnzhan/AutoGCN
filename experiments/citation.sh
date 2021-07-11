#!/bin/sh
gpu_id=0
cd ..
mkdir baseline_citation
for model in  AUTOGCN ChebNet GatedGCN GraphSage MLP GCN GIN GAT MoNet; do
    for data in PUBMED;do
        python main_CitationGraphs_node_classification.py --runs 5 --gpu_id $gpu_id --config './configs/CitationGraphs_node_classification_'$model'.json' --dataset $data > baseline_citation/$model-$data.log
done
done