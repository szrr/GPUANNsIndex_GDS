# GANNS
## Introduction
This project includes (1) a GPU-based algorithm GANNS which can accelerate 
the ANN search on proximity graphs by re-designing the classical CPU-based search algorithm 
and using GPU-friendly data structures. 
(2) novel GPU-based proximity graph construction algorithms which ensure the quality of the resulting proximity graph.



### Search
To use search algorithm, generate query instance.
```zsh
./generate_query_instances.sh [dim] [metric]
```
For instance, the following command line generates an executable program which can work on datasets with dimension 128 and metric euclidean distance. 
```zsh
./generate_query_instances.sh 128 l2
```
Currently, we support dimension ```no larger than 960``` and three metrics: ```euclidean distance (l2)```, ```cosine similarity (cos)``` and ```inner product (ip)```.

To use the generated executable program, the following parameters need to be provided.
```zsh
./query_128_l2 [base_path] [query_path] [graph_path] [groundtruth_path] [e] [k] [points_num]
```
Specifically, ```[base_path]``` is the directory of data points (database); ```[query_path]``` is the directory of query points; 
```[groundtruth_path]``` is the directory of groundtruth; ```[e]``` represents the number of explored vertices; ```[k]``` denotes the number of returned nearest neighbors;```[points_num]```is the points number of the dataset; 

For instance, the following command line performs ANN search on NSW constructed on SIFT dataset 
(These files in the command line are not included in this project due to their size).
```zsh
./query_128_l2 ../dataset/sift/base.fvecs ../dataset/sift/query.fvecs ../dataset/sift/base.fvecs_64_16.nsw ../dataset/sift/groundtruth.ivecs 64 10 1000000
```

### Construction
To use construction algorithm, generate build instance.
```zsh
./generate_build_instances.sh [dim] [metric]
```
Similarly, we support dimension ```no larger than 960``` and three metrics: ```euclidean distance (l2)```, ```cosine similarity (cos)``` and ```inner product (ip)```.

To use the generated executable program, the following parameters need to be provided.
```zsh
./build_128_l2 [base_path] [e] [d_min] [points_num]
```
Specifically, ```[base_path]``` is the directory of data points (database);  ```[e]``` represents the number of explored vertices; ```[d_min]``` denotes minimum degree in the proximity graph (by default, d_max = 2 * d_min);```[points_num]```is the points number of the dataset;

For instance, the following command line establishes a HNSW graph on SIFT dataset 
(These files in the command line are not included in this project due to their size).
```zsh
./build_128_l2 ../dataset/sift/base.fvecs 64 16 1000000
```
Notice the parameters ```dim``` and ```metric``` that are provided to generate query (resp. build) instances must be consistent with dimension and metric of datasets. 
Otherwise, the executable program may shut down, or the recall (resp. the quality of proximity graph) may be poor.

## Datasets
Base: /mnt/data1/szr/dataset/sift1b/bigann_base.bvecs          the size of bigann_base.bvecs is 1 billion
Query: /mnt/data1/szr/dataset/sift1b/bigann_query.bvecs
Ground Truth /mnt/data1/szr/dataset/sift1b/gnd/idx_'x'M.ivecs  'x' is the size of dataset like /mnt/data1/szr/dataset/sift1b/gnd/idx_5M.ivecs
