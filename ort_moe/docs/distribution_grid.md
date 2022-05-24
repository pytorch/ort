#DistributionGrid 

DistributionGrid provides simple interface to create and manage process groups for 
various distributed training configurations. 

The grid creates process groups, using torch.distributed, based on distribution scheme
requested. Currently, Data Parallelism, Expert Parallelism,  Expert Parallel Replica and
Expert Slicing techniques are supported. Support for Pipeline Parallelism is in early stage.
The grid provides interface to query process group, world size and rank information. 

The grid also provides interface to map local expert id to global expert id and vice versa 
for the models using Mixture Of Experts technique.

# Examples
    [1] Create standard data parallel grid where each rank holds entire copy of the model.
        dgrid = DistributionGrid(data_parallel_group_size = <no_of_ranks_available>)
        pg = dgrid.get_data_parallel_group()
        ws = dgrid.get_data_parallel_world_size()
        rank = dgird.get_data_parallel_rank()

    [2] Create expert parallel grid where experts are evenly distributed among available ranks.
        dgrid = DistributionGrid(expert_parallel_group_size = <no_of_ranks_available>)
        pg = dgrid.get_expert_parallel_group()
        ws = dgrid.get_expert_parallel_world_size()
        rank = dgird.get_expert_parallel_rank()

        The grid also provide interface to create process groups for MoE layer while applying ZeRO technique.
        mgroup = dgrid.get_moe_group()

    [3] Create expert slicing grid where each expert are evenly sharded among avaiable ranks.
        dgrid = DistributionGrid(expert_slicing_group_size = <no_of_ranks_available>)

    [4] Create replicas of expert parallel or expert slicing distributions 
        dgrid = DistributionGrid(expert_parallel_group_size (or expert_slicing_group_size) = <no_of_ranks_used_in_single_expert_parallel_copy>),
            expert_replica_group_size = <no_of_expert_replicas>)

    [5] Initialize pipeline parallel distribution
        dgrid = DistributionGrid(num_of_pipeline_stage = 4)
        ncount = dgrid.get_num_of_nodes_in_pipeline()
        stage_count = dgrid.get_num_of_pipeline_stages()
        first_device = dgrid.get_first_pipeline_stage_device()
        last_device = dgrid.get_last_pipeline_stage_device()
        
