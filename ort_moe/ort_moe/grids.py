# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import math
import torch
import torch.distributed as dist


class BaseGrid:
    '''BaseGrid class provides an abstract class in the case that no distributed backend is available.
    '''
    def __init__(self):
        self._EXPERT_REPLICA_GROUP = None
        self._EXPERT_PARALLEL_GROUP = None
        self._EXPERT_SLICING_GROUP = None
        self._export_relocation_map = {}

    def get_expert_replica_group(self):
        return self._EXPERT_REPLICA_GROUP

    '''
    Interface to support Expert Parallelism.

    In this distribution technique, experts in each layer are evenly distributed among available ranks
    while non-expert parameters are replicated on each rank.
    '''
    def get_expert_parallel_group(self):
        return self._EXPERT_PARALLEL_GROUP

    def get_expert_parallel_world_size(self):
        if self.get_expert_parallel_group() is not None:
            return dist.get_world_size(group=self.get_expert_parallel_group())
        return 1

    def get_expert_parallel_rank(self):
        if self.get_expert_parallel_group() is not None:
            return dist.get_rank(group=self.get_expert_parallel_group())
        return 0

    '''
    Interface to support Expert Slicing.

    In this distribution technique, experts in each layer are sliced on hidden dimension and
    sharded across available ranks. The non-expert parameters are replicated on each rank.
    '''
    def get_expert_slicing_group(self):
        return self._EXPERT_SLICING_GROUP

    '''
    Simplified common interface for Expert Parallel and Expert Slicing
    '''
    def get_expert_group(self):
        if self._EXPERT_PARALLEL_GROUP is None:
            return self._EXPERT_SLICING_GROUP
        else:
            return self._EXPERT_PARALLEL_GROUP

    def get_expert_world_size(self):
        if self.get_expert_slicing_group() is not None:
            return dist.get_world_size(group=self.get_expert_slicing_group())
        elif self.get_expert_parallel_group() is not None:
            return dist.get_world_size(group=self.get_expert_parallel_group())
        return 1

    def get_expert_rank(self):
        if self.get_expert_parallel_group() is not None:
            return dist.get_rank(group=self.get_expert_parallel_group())
        elif self.get_expert_slicing_group() is not None:
            return dist.get_rank(group=self.get_expert_slicing_group())
        return 0

    def get_mpi_group_for_expert_group(self):
        return None


class DistributionGrid(BaseGrid):
    '''DistributionGrid provides simple interface to create and manage process groups
    for various distributed training configurations.

    [1] Create standard data parallel grid where each rank holds entire copy of the model.
        dgrid = DistributionGrid(data_parallel_group_size = <no_of_ranks_available>)

    [2] Create expert parallel grid where experts are evenly distributed among available ranks.
        dgrid = DistributionGrid(expert_parallel_group_size = <no_of_ranks_available>)

    [3] Create expert slicing grid where each expert are evenly sharded among avaiable ranks.
        dgrid = DistributionGrid(expert_slicing_group_size = <no_of_ranks_available>)

    [4] Create replicas of expert parallel or expert slicing distributions
        dgrid = DistributionGrid(expert_parallel_group_size (or expert_slicing_group_size) = <no_of_ranks_used_in_single_expert_parallel_copy>),
            expert_parallel_replica_group_size = <no_of_expert_replicas>)

    [5] Initialize pipeline parallel distribution
        dgrid = DistributionGrid(num_of_pipeline_stage = 4)

    Arguments:
        data_parallel_group_size: number of data parallel copies of the model
        expert_parallel_group_size: number of GPUs sharing experts of single layer
        expert_slicing_group_size: number of GPUs experts are sharded onto of single layer
        expert_parallel_replica_group_size: number of data parallel copies of experts
        num_of_nodes_in_pipeline: number of nodes used in pipeline parallel mode
        num_of_pipeline_stage: number of pipeline stages
        options: Various grid options
    '''
    def __init__(self, data_parallel_group_size = None, expert_parallel_group_size = None,
                 expert_parallel_replica_group_size = None, expert_slicing_group_size = None,
                 num_of_nodes_in_pipeline = None, num_of_pipeline_stage = None, options = {}):
        #print("==> initialize dgrid")
        super().__init__()
        self._DATA_PARALLEL_GROUP = None
        # base rank for broadcasting initialized weights
        self._EXPERT_REPLICA_GROUP_BCAST_SRC_RANK = None
        self._ep_rank_list = None
        self._es_rank_list = None
        self._dp_rank_list = None
        self._MPI_EP_GROUP = None
        self._MPI_ES_GROUP = None
        # Used to wrap MoE layer by FullyShardedDataParallel for the experts, which are not sharded
        self._MOE_GROUP = None

        self._NUM_PIPELINE_STAGES = 0
        self._NUM_NODES_IN_PIPELINE = 0

        #TODO: Temporarily assign expert_parallel_group_size to expert_replica_group_size, later when introduce the expert slicing, remove this
        # and change the API to expert_replica
        expert_replica_group_size = expert_parallel_replica_group_size

        # Sanity check
        if data_parallel_group_size is None and expert_parallel_group_size is None \
            and expert_replica_group_size is None and expert_slicing_group_size is None \
            and num_of_nodes_in_pipeline is None and num_of_pipeline_stage is None:
            # Did not request any distribution strategy
            return

        self._options = options
        assert not(expert_slicing_group_size is not None and expert_parallel_group_size is not None), \
            "Cannot have both expert slicing and expert parallel"
        is_expert_slicing = expert_slicing_group_size is not None
        expert_group_size = expert_slicing_group_size if is_expert_slicing else expert_parallel_group_size

        # Standard data parallel distribution without any separate distribution for experts
        if data_parallel_group_size is not None:

            assert expert_parallel_group_size is None, \
                "Standard Data Parallelism with Expert Parallelism is not supported"
            assert expert_slicing_group_size is None, \
                "Standard Data Parallelism with Expert Slicing is not supported"
            assert expert_replica_group_size is None, \
                "Standard Data Parallelism with Expert Replicas is not supported"
            assert num_of_nodes_in_pipeline is None and num_of_pipeline_stage is None, \
                "Standard Data Parallelism with Pipeline Parallelism is not supported"

            self._initialize_data_parallel_group(data_parallel_group_size)
            return

        # Standard pipeline parallelism
        if num_of_pipeline_stage is not None and num_of_nodes_in_pipeline is not None:

            assert data_parallel_group_size is None, \
                "Standard Data Parallelism with Pipeline Parallelism is not supported"
            assert expert_group_size is None, \
                "Expert Parallelism/Slicing  with Expert Replicas is not supported"
            assert expert_replica_group_size is None, \
                "Pipeline Parallelism with Expert Repliacs is not supported"

            self._initialize_pipeline_parallel(num_of_nodes_in_pipeline, num_of_pipeline_stage)
            return

        # Expert parallel replica distribution
        if expert_replica_group_size is not None and expert_replica_group_size > 1:

            assert data_parallel_group_size is None, \
                "Standard Data Parallelism with Expert Replicas is not supported"
            assert expert_group_size is not None, \
                "Expert Replicas require Expert Parallelism or Expert Slicing"
            assert num_of_nodes_in_pipeline is None and num_of_pipeline_stage is None, \
                "Expert Replicas with Pipeline Parallelism is not supported"

            self._initialize_expert_parallel_or_expert_slicing_replica_groups(expert_group_size, expert_replica_group_size, is_expert_slicing)
            return

        # Standard expert parallel distribution or expert slicing distribution
        if expert_group_size is not None:

            assert data_parallel_group_size is None, \
                "Expert Parallelism/Slicing with Standard Data Parallelism is not supported"
            assert num_of_nodes_in_pipeline is None and num_of_pipeline_stage is None, \
                "Expert Parallelism/Slicing with Pipeline Parallelism is not supported"

            self._initialize_expert_parallel_or_slicing_group(expert_group_size, is_expert_slicing)
            return

        return

    def _initialize_pipeline_parallel(self, num_nodes, nstages):
        '''
        Initialize pipelining framework.
        Args:
          nstages : Total number of pipeline stages.
        '''
        #print("==> inside initialize-pipeline_parallel :")
        assert num_nodes == 1, "Only single node pipeline parallelism is supported right now"
        from torch.distributed import rpc
        import tempfile
        tmpfile = tempfile.NamedTemporaryFile()
        rpc.init_rpc(
            name="grid_worker",
            rank=0,
            world_size=num_nodes,
            rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
                init_method="file://{}".format(tmpfile.name)
            )
        )
        self._NUM_PIPELINE_STAGES = nstages
        self._NUM_NODES_IN_PIPELINE = num_nodes

    def _initialize_data_parallel_group(self, dp_ranks):
        '''
        Initialize new process group for data parallel distribution.
        Args:
          dp_ranks : Total number of ranks used to replicate the model otherwise
                     torch.distribued world_size is used.
        '''
        #print("==> inside initialize_expert_parallel_group :")
        if dist.is_initialized() is False:
            self._initialize_torch_distributed()

        self._DATA_PARALLEL_GROUP = dist.new_group(range(0,dp_ranks))

    def _initialize_expert_parallel_or_slicing_group(self, ranks_count, is_expert_slicing = False):
        '''
        Initialize new process group for expert parallel distribution or expert slicing distribution
        Args:
          ranks_count : Total number of ranks used to distribute experts parallely otherwise
                     torch.distribued world_size is used.
          is_expert_slicing : True if using expert slicing, otherwise use expert parallelism
        '''
        # In expert parallel distribution, the non-expert parameters are replicated
        # using standard data parallel distribution. The expert parameters are divided among
        # available ranks, a form of model parallelism.

        #print("==> inside initialize_expert_parallel_group :")
        if dist.is_initialized() is False:
            self._initialize_torch_distributed()

        rank_list = list(range(0, dist.get_world_size()))
        rank_list = [rank_list[i:i+ranks_count] for i in range(0, len(rank_list), ranks_count)]
        if ranks_count == dist.get_world_size():
            parallel_group = dist.group.WORLD
        else:
            #if local rank in the sublist of the rank list, then the parallel group is composed of the sublist
            parallel_group,_ = self._build_process_group(rank_list, dist.get_rank())
        assert parallel_group is not None
        if is_expert_slicing:
            self._es_rank_list = rank_list
            self._EXPERT_SLICING_GROUP = parallel_group
        else:
            self._ep_rank_list = rank_list
            self._EXPERT_PARALLEL_GROUP = parallel_group

    def _initialize_expert_parallel_or_expert_slicing_replica_groups(self, expert_ranks, dp_ranks, is_expert_slicing = False):
        '''
        Initialize new process groups for exper parallel replicas.
        Let's say we have a total of 16 GPUs denoted by g0 ... g15 and we
        want 2-data parallel copies of each experts.

        This function will create 8 expert replica groups, 2 expert parallel groups as:
        8 expert replica groups:
        [g0, g8], [g1, g9], [g2, g10], [g3, g11], [g4, g12], [g5, g13], [g6, g14], [g7, g15]
        2  expert parallel groups or expert slicing group:
        [g0, g1, g2, g3, g4, g5, g6, g7], [g8, g9, g10, g11, g12, g13, g14, g15]

        Note that for efficiency of all2all, the caller should make sure adjacent ranks
        are on the same expert parallel group.

        Args:
          expert_ranks : Total number of ranks used to distribute expert parallely in one replica.
          dp_ranks : Total number of expert parallel replicas.
        '''

        if dist.is_initialized() is False:
            self._initialize_torch_distributed()

        assert dist.get_world_size() % dp_ranks == 0, "Only support the n-way expert parallel replicas when world size is dividable by n"
        assert dp_ranks > 1, "There should be at least 2 replicas"

        rank_schedule = self._options.get("rank_schedule", "row_major")
        if rank_schedule == "row_major":
            row_major_list = True
        elif rank_schedule == "column_major":
            row_major_list = False
        else:
            assert False, "Selected grid rank schedule is not supported"
        is_replica_in_same_node = self._options.get("is_replica_in_same_node", False)
        assert row_major_list or not is_replica_in_same_node, "column_major with is_replica_in_same_node == True is not supported"

        def rearrange(rlist, eranks):
            rearranged =[]
            for j in range(0, eranks):
                for i in range(j, len(rlist), eranks):
                    rearranged.append(i)
            return rearranged
        rank_list = list(range(0, dist.get_world_size()))
        rearranged_list = rearrange(rank_list, expert_ranks)

        expert_rank_list = []
        if row_major_list is True:
            if is_replica_in_same_node:
                expert_rank_list = [rank_list[i::dp_ranks] for i in range(0, dp_ranks)]
                self._dp_rank_list = [rank_list[i:i+dp_ranks] for i in range(0, len(rank_list), dp_ranks)]
            else:
                expert_rank_list = [rank_list[i:i+expert_ranks] for i in range(0, len(rank_list), expert_ranks)]
                self._dp_rank_list = [rank_list[i::expert_ranks] for i in range(0, expert_ranks)]
        else:
            expert_rank_list = [rearranged_list[i:i+expert_ranks] for i in range(0, len(rearranged_list), expert_ranks)]
            self._dp_rank_list = [rearranged_list[i::expert_ranks] for i in range(0, expert_ranks)]

        # [1] Setup process group for expert parallel distribution
        local_rank = dist.get_rank()
        exp_group,_ = self._build_process_group(expert_rank_list, local_rank)
        if is_expert_slicing:
            self._es_rank_list = expert_rank_list
            self._EXPERT_SLICING_GROUP = exp_group
        else:
            self._ep_rank_list = expert_rank_list
            self._EXPERT_PARALLEL_GROUP = exp_group

        # [2] Setup process group for expert parallel replicas
        self._EXPERT_REPLICA_GROUP, self._EXPERT_REPLICA_GROUP_BCAST_SRC_RANK = self._build_process_group(self._dp_rank_list, local_rank)

        self._DATA_PARALLEL_GROUP = dist.group.WORLD

    '''
    torch.distributed process group routines.
    '''
    def _initialize_torch_distributed(self):
        '''
        Initialize torch.distributed process group using NCCL as backend.
        '''
        env_dict = {
            key: os.environ[key]
            for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
        }
        print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
        dist.init_process_group(backend="nccl")
        print(
            f"[{os.getpid()}] world_size = {dist.get_world_size()}, "
            + f"rank = {dist.get_rank()}, backend={dist.get_backend()}"
        )

    def cleanup(self):
        '''
        Clean up the grid. If process group is initialized then distroy it.
        '''
        if dist.is_initialized():
            dist.destroy_process_group()

    '''
    Helper routines
    '''
    def _build_process_group(self, rank_lists, rank):
        '''
        Build new process groups and return one for the current rank
        Args:
            rank_lists : List of ranks. New group will created for each item in the list.
            rank : The group corresponding to the rank will be returned.
        Return Value:
            The process group of the input rank and the src rank of the process group.
        '''
        assert rank is not None
        assert rank_lists is not None
        pg = None
        min_rank = None
        # Each rank in main group need to go through each new_group() function even if it doesn't belong to that group
        # https://pytorch.org/docs/stable/_modules/torch/distributed/distributed_c10d.html#new_group
        for r in rank_lists:
            tmp = dist.new_group(r)
            if rank in r:
                pg = tmp
                min_rank = min(r)
        return pg, min_rank

    def get_moe_group(self):
        '''
        This group is used by FullyShardedDataParallel (FSDP) to wrap MoE layers.
        '''
        assert self.get_expert_parallel_group() is not None, "Unsupported expert parallel configuration"
        assert self.get_expert_replica_group() is None, "Unsupported expert replica configuration"
        assert self.get_expert_slicing_group() is None, "Unsupported expert slicing configuration"
        assert self.get_num_of_nodes_in_pipeline() == 0, "Unsupported expert pipeline configuration"
        assert self.get_num_of_pipeline_stages() == 0, "Unsupported expert pipeline configuration"
        if self._MOE_GROUP is not None:
            return self._MOE_GROUP

        moe_groups = [[i] for i in range(dist.get_world_size())]
        local_rank = dist.get_rank()
        self._MOE_GROUP,_ = self._build_process_group(moe_groups, local_rank)
        return self._MOE_GROUP

    def get_expert_slicing_world_size(self):
        if self.get_expert_slicing_group() is not None:
            return dist.get_world_size(group=self.get_expert_slicing_group())
        return 1

    def get_expert_slicing_rank(self):
        if self.get_expert_slicing_group() is not None:
            return dist.get_rank(group=self.get_expert_slicing_group())
        return 0

    '''
    Interface to support Expert replications.

    In this distribution technique, expert parallel distribution is replicated multiple times.
    '''
    def get_expert_parallel_replica_group(self):
        return None if self.get_expert_parallel_group() is None else self.get_expert_replica_group()

    def get_expert_parallel_replica_world_size(self):
        return 1 if self.get_expert_parallel_group() is None else self.get_expert_replica_world_size()

    def expert_parallel_replica_group_member_rank_lists(self):
        return self._dp_rank_list

    def get_expert_parallel_replica_rank(self):
        return 0 if self.get_expert_parallel_group() is None else self.get_expert_replica_rank()

    def get_expert_parallel_replica_src_rank(self):
        return None if self.get_expert_parallel_group() is None else self.get_expert_replica_src_rank()

    def get_mpi_group_for_expert_parallel_group(self):
        return None if self.get_expert_parallel_group() is None else self.get_mpi_group_for_expert_group()

    def get_expert_replica_world_size(self):
        if self.get_expert_replica_group() is not None:
            return dist.get_world_size(group=self.get_expert_replica_group())
        return 1

    def get_expert_replica_rank(self):
        if self.get_expert_replica_group() is not None:
            return dist.get_rank(group=self.get_expert_replica_group())
        return 0

    def get_expert_replica_src_rank(self):
        return self._EXPERT_REPLICA_GROUP_BCAST_SRC_RANK

    # Create MPI group for correspending expert parallel group or expert slicing gorup.
    def get_mpi_group_for_expert_group(self):
        is_expert_slicing = self._es_rank_list is not None
        if self._MPI_EP_GROUP is not None:
            return self._MPI_ES_GROUP if is_expert_slicing else self._MPI_EP_GROUP
        from mpi4py import MPI
        expert_rank_list = self._es_rank_list if is_expert_slicing else self._ep_rank_list
        if expert_rank_list is None:
            return None
        expert_group = dist.group.WORLD
        for g in expert_rank_list:
            tmp = MPI.COMM_WORLD.Create_group(MPI.COMM_WORLD.group.Incl(g))
            if MPI.COMM_WORLD.Get_rank() in g:
                expert_group = tmp
        if is_expert_slicing:
            self._MPI_ES_GROUP = expert_group
        else:
            self._MPI_EP_GROUP = expert_group
        return expert_group

    '''
    Interface to support Data Parallel.

    In this standard distribution technique, all model parameters are replicated on each rank.
    '''
    def get_data_parallel_group(self):
        #A workaround to set _DATA_PARALLEL_GROUP in EP/ES case
        #If the expert_group is WORLD, initiate a new group with the world size.
        #Otherwise, we set the expert_group() as the data_parallel group
        if self.get_expert_group() is not None and self._DATA_PARALLEL_GROUP is None:
            self._DATA_PARALLEL_GROUP = self.get_expert_group()
        return self._DATA_PARALLEL_GROUP

    def get_data_parallel_world_size(self):
        return dist.get_world_size(group=self.get_data_parallel_group())

    def get_data_parallel_rank(self):
        return dist.get_rank(group=self.get_data_parallel_group())

    '''
    Interace to support Pipeline Parallelism

    In this distribution technique, a form of model parallelism, model parameters are distributed
    on multiple rank such that activations from previous rank is sent to next rank as an input. The
    model is divided into stages. Currently, only one stage per rank is supported.
    '''
    def get_num_of_nodes_in_pipeline(self):
        return self._NUM_NODES_IN_PIPELINE

    def get_num_of_pipeline_stages(self):
        return self._NUM_PIPELINE_STAGES

    def get_first_pipeline_stage_device(self):
        # FIXME : Sometimes the pipeline may start at rank 4 and continue for next 2 ranks.
        return 0

    def get_last_pipeline_stage_device(self):
        # FIXME : Return torch device here ?
        return self._NUM_PIPELINE_STAGES - 1

    '''
    Helper routines to help map experts from one distribution scheme to another.
    '''
    def map_expert_id_local_to_global(self, total_experts, eparm_id):
        '''
        Map local expert parameter to a global index.
        Args:
            total_experts : Total number of experts managed by this grid.
            eparam_id : Local ID of the expert parameter that will be mapped to a global ID based
                        on the grid's distribution strategy.
        Return Values:
            Global ID of this local expert parameter
        '''
        assert self.get_expert_parallel_group() is not None, "Unsupported expert mapping configuration"
        assert self.get_expert_slicing_group() is None, "Unsupported expert mapping configuration"
        assert self.get_num_of_nodes_in_pipeline() == 0, "Unsupported expert mapping configuration"
        assert self.get_num_of_pipeline_stages() == 0, "Unsupported expert mapping configuration"

        ep_world_size = self.get_expert_parallel_world_size()
        assert total_experts % ep_world_size == 0, 'Experts can not be evenly divided among ranks'
        assert int(total_experts/ep_world_size) > eparm_id, 'Local expert id out of range'
        ep_rank = self.get_expert_parallel_rank()
        # Calculate global id of the expert
        gid = eparm_id + int(total_experts/ep_world_size) * ep_rank

        # If gid was relocated then use the original global id
        if gid in self._export_relocation_map:
            gid = self._export_relocation_map[gid]

        if self.get_expert_parallel_replica_group() is None:
            return gid
        else:
            return gid, self.get_expert_parallel_replica_rank()

    def map_expert_id_global_to_local(self, total_experts, global_eparm_id):
        '''
        Map globally indexed expert parameter locally based on current distribution scheme.
        Args:
            total_experts : Total number of experts managed by this grid.
            global_eparam_id : Global ID of the expert parameter that will be mapped to a local ID based
                               on the grid's distribution strategy.
        Return Values:
            nrank, nid : Local rank and ID of this global expert parameter will be mapped to
        '''
        assert self.get_expert_parallel_group() is not None, "Unsupported expert mapping configuration"
        assert self.get_expert_slicing_group() is None, "Unsupported expert mapping configuration"
        assert self.get_num_of_nodes_in_pipeline() == 0, "Unsupported expert mapping configuration"
        assert self.get_num_of_pipeline_stages() == 0, "Unsupported expert mapping configuration"
        assert global_eparm_id < total_experts, 'Global expert id out of range'

        # If global param id was relocated then use the original global id
        if global_eparm_id in self._export_relocation_map:
            global_eparm_id = self._export_relocation_map[global_eparm_id]

        # Map global id to the  grid
        ep_world_size = self.get_expert_parallel_world_size()
        assert total_experts % ep_world_size == 0, 'Experts can not be evenly divided among ranks'
        ep_rank_size = int(total_experts / ep_world_size)
        nrank =  math.floor(global_eparm_id / ep_rank_size)
        nid = global_eparm_id - ep_rank_size * nrank
        if self.get_expert_parallel_replica_group() is None:
            return nrank, nid

        # expert parallel replica
        result = [(nrank, nid)]
        for i in range(1, self.get_expert_parallel_replica_world_size()):
            result += [(nrank + i * self.get_expert_parallel_world_size(), nid)]
        return result

    def exchange_expert_location(self, global_expert_id1, global_expert_id2):
        '''
        Exchange expert location across ranks. This can be used to balance token routing.
        Args:
            global_expert_id1, global_expert_id2 : Two experts that will be exchanged across ranks.
        '''
        if global_expert_id1 == global_expert_id2:
            return
        self._export_relocation_map[global_expert_id1] = global_expert_id2
        self._export_relocation_map[global_expert_id2] = global_expert_id1
        return

    def remove_expert_relocation(self, id):
        '''
        Remove expert relocation entry from the relocation map.
        Args:
            id : Expert id that will be removed from the relocation map.
        '''
        if id in self._export_relocation_map:
            id2 = self._export_relocation_map[id]
            del self._export_relocation_map[id]
            del self._export_relocation_map[id2]

    def get_get_relocation_id(self, id):
        '''
        If the expert is relocated then return its relocated expert id.
        Args:
            id : Expert id whose relocated id will be returned.
        '''
        return self._export_relocation_map.get(id, None)

    def get_expert_relocation_map(self):
        return self._export_relocation_map

    def set_expert_relocation_map(self, map):
        self._export_relocation_map = map