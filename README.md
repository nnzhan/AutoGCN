# AutoGCN


<br>

## 1. Benchmark installation

[Follow these instructions](./docs/01_benchmark_installation.md) to install the benchmark and setup the environment.


<br>

## 2. Download datasets

[Proceed as follows](./docs/02_download_datasets.md) to download the benchmark datasets.


<br>

## 3. Training


```
gpu_id=0

model=AUTOGCN

python main_superpixels_graph_classification.py --config './configs/superpixels_graph_classification_'$model'_MNIST.json' --runs 5 --gpu_id $gpu_id

python main_superpixels_graph_classification.py --config './configs/superpixels_graph_classification_'$model'_CIFAR10.json' --runs 5 --gpu_id $gpu_id

python main_SBMs_node_classification.py --config './configs/SBMs_node_clustering_'$model'_CLUSTER.json' --runs 5 --gpu_id $gpu_id

python main_SBMs_node_classification.py --config './configs/SBMs_node_clustering_'$model'_PATTERN.json' --runs 5 --gpu_id $gpu_id

python main_molecules_graph_regression.py --config './configs/molecules_graph_regression_'$model'_ZINC.json' --runs 5 --gpu_id $gpu_id

```


<br>

## 4. Testing

```
gpu_id=0

python test_superpixel.py --config './configs/superpixels_graph_classification_AUTOGCN_MNIST.json' --model_path ./pretrained_models/AUTOGCN-MNIST.pkl --gpu_id $gpu_id

python test_superpixel.py --config './configs/superpixels_graph_classification_'AUTOGCN'_CIFAR10.json' --model_path ./pretrained_models/AUTOGCN-CIFAR.pkl --gpu_id $gpu_id

python test_sbm.py --config './configs/SBMs_node_clustering_AUTOGCN_CLUSTER.json' --model_path ./pretrained_models/AUTOGCN-CLUSTER.pkl --gpu_id $gpu_id

python test_sbm.py --config './configs/SBMs_node_clustering_AUTOGCN_CLUSTER.json' --model_path ./pretrained_models/AUTOGCN-PATTERN.pkl --gpu_id $gpu_id

python test_molecule.py --config './configs/molecules_graph_regression_AUTOGCN_ZINC.json' --model_path ./pretrained_models/AUTOGCN-ZINC.pkl --gpu_id $gpu_id

```


<br>

## 5. Ablation Study

```
cd experiments
sh autogcn-ab-cluster.sh
sh autogcn-ab-molecule.sh

```

## 6. Other experments

Other experiment scripts can be found at ./experiments folder

<br>

## 7. Acknowledgement

We thank Dwivedi et al. for providing the gnn benchmarking framework. Original repository of the gnn benchmarking framework can be found at [https://github.com/graphdeeplearning/benchmarking-gnns](https://github.com/graphdeeplearning/benchmarking-gnns).

Reference

```
@article{dwivedi2020benchmarkgnns,
  title={Benchmarking Graph Neural Networks},
  author={Dwivedi, Vijay Prakash and Joshi, Chaitanya K and Laurent, Thomas and Bengio, Yoshua and Bresson, Xavier},
  journal={arXiv preprint arXiv:2003.00982},
  year={2020}
}
```






<br><br><br>

