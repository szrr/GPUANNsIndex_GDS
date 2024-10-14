# Giann_core

> giann_core is a key component built on NVIDIA BaM that provides GPU direct access to SSD for approximate nearest neighbor search. It is inspired by GIDS and GMT, and provides the ability to use host memory as another level of cache to reduce read latency.



## References
The source code of this repository comes from https://zenodo.org/records/10873493, and the corresponding paper is **GMT: GPU Orchestrated Memory Tiering for the Big Data Era**.


GMT is built on top of BaM, please refer to [BaM](https://github.com/ZaidQureshi/bam) for installation.the corresponding paper is **GPU-Initiated On-Demand High-Throughput Storage Access in the BaM System Architecture**.

Our project also refers to the paper **GIDS: Accelerating Sampling and Aggregation Operations in GNN Frameworks with GPU Initiated Direct Storage Accesses**