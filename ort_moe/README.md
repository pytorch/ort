# Introduction
Mixture Of Experts (MoE) implementation in PyTorch
This repo contains following components
#moe_module
moe_module  contains PyTorch implementation of MoE, Experts and Gates and Transformer layers. This module is used by 1P workloads to inject MoE layers in their model. We aim to implement moe_module such that it is amenable to variety of model distribution techniques for large scale (100B+ param) training.
#Proxy Models
We have implemented proxy of Gshard, Switch Transformer etc.. models using moe_module in moe_models.py. These models serves two purposes. First to simple standalone proxy model for approximate performance analysis and characterization of scaling efforts. Second is to evaluate flexibility of moe_module interface in variety of situations. We encourage contribution of new proxy models.
#Trainer
We have extended NLP trainer from Pytorch tutorial in baseline_nlp.py to run proxy models and collect performance data. We use WIkiText as a dataset, which is obviously not representative of real world 1P workload scenarios.  The trainer allows us to experiment with variety of PyTorch packages/techniques such as DeepSpeed, Apex, torch DistributedDataParallel etc.. We welcome contributions to incorporate Pipeline Parallelism and Megatron-style training.
# ITP Scripts
We have scripts available in experiments/itp folder to easily launch jobs on ITP cluster. We have two ready made experiments available, Switch-CA and Switch-CB. They are two variants of Switch Transformer model scaled to 100B parameter size.

# Updates
0.1.7 : Adapt new name - ort_moe

