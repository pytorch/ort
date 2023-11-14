# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from mpi4py import MPI
import torch.distributed as dist
import torch
import pytest
import os
import tempfile

from ort_moe.grids import DistributionGrid

assert torch.cuda.is_available()

BACKEND = dist.Backend.NCCL
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29501"  # torch 1.5 compatibility
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if not dist.is_initialized():
    dist.init_process_group(backend=BACKEND, rank=rank, world_size=size)

def test_setup(device = rank):
    assert dist.get_world_size() == 4

def test_empty_grid(device = rank):
    if dist.get_world_size() != 4:
        return
    dgrid = DistributionGrid()
    assert dgrid.get_data_parallel_group() is None
    assert dgrid.get_expert_parallel_group() is None
    assert dgrid.get_expert_replica_group() is None
    assert dgrid.get_expert_parallel_replica_group() is None

def test_data_parallel_grid(device = rank):
    if dist.get_world_size() != 4:
        return
    dgrid = DistributionGrid(data_parallel_group_size = 4)
    assert dgrid.get_data_parallel_group() is not None
    print("====> ", dgrid.get_data_parallel_world_size())
    assert dgrid.get_data_parallel_world_size() == 4
    assert dgrid.get_data_parallel_rank() == rank
    assert dgrid.get_expert_parallel_group() is None
    assert dgrid.get_expert_replica_group() is None
    assert dgrid.get_expert_parallel_replica_group() is None

def test_expert_parallel_grid(device = rank):
    if dist.get_world_size() != 4:
        return
    dgrid = DistributionGrid(expert_parallel_group_size = 4)
    assert dgrid.get_expert_parallel_group() is not None
    assert dgrid.get_expert_parallel_world_size() == 4
    assert dgrid.get_expert_world_size() == 4
    assert dgrid.get_expert_replica_group() is None
    assert dgrid.get_expert_parallel_replica_group() is None
    assert dgrid.get_expert_replica_world_size() == 1

def test_expert_slicing_grid(device = rank):
    if dist.get_world_size() != 4:
        return
    dgrid = DistributionGrid(expert_slicing_group_size = 4)
    assert dgrid.get_expert_slicing_group() is not None
    assert dgrid.get_expert_slicing_world_size() == 4
    assert dgrid.get_expert_world_size() == 4
    assert dgrid.get_expert_replica_group() is None
    assert dgrid.get_expert_parallel_replica_group() is None

def test_expert_replica_grid(device = rank):
    if dist.get_world_size() != 4:
        return
    dgrid = DistributionGrid(expert_parallel_group_size = 2, expert_parallel_replica_group_size = 2)
    assert dgrid.get_expert_parallel_group() is not None
    assert dgrid.get_expert_parallel_replica_group() is not None
    assert dgrid.get_expert_replica_world_size() == 2
    assert dgrid.get_expert_parallel_replica_world_size() == 2
    assert dgrid.get_expert_parallel_world_size() == 2
    assert dgrid.get_expert_world_size() == 2

def test_expert_replica_grid_2(device = rank):
    if dist.get_world_size() != 4:
        return
    dgrid = DistributionGrid(expert_parallel_group_size = 4, expert_parallel_replica_group_size = 1)
    assert dgrid.get_expert_parallel_replica_rank() == 0

def test_pipeline_parallel_grid(device = rank):
    if dist.get_world_size() != 4:
        return
    dgrid = DistributionGrid(num_of_nodes_in_pipeline = 1, num_of_pipeline_stage = 4)
    assert dgrid.get_expert_parallel_group() is None
    assert dgrid.get_expert_parallel_replica_group() is None
    assert dgrid.get_expert_slicing_group() is  None
    assert dgrid.get_num_of_nodes_in_pipeline() == 1
    assert dgrid.get_num_of_pipeline_stages() == 4
    assert dgrid.get_first_pipeline_stage_device() == 0
    assert dgrid.get_last_pipeline_stage_device() == 3

def test_expert_mapping_2(device = rank):
    if dist.get_world_size() != 4:
        return

    dgrid = DistributionGrid(expert_parallel_group_size = 4)
    nrank, nid = dgrid.map_expert_id_global_to_local(64, 30)
    assert nrank == 1 and nid == 14

    grank = dgrid.map_expert_id_local_to_global(64, 14)
    if rank == 0:
        assert grank == 14
    elif rank == 1:
        assert grank == 30
    elif rank == 2:
        assert grank == 46
    elif rank == 3:
        assert grank == 62

def test_expert_mapping_3(device = rank):
    if dist.get_world_size() != 4:
        return

    dgrid = DistributionGrid(expert_parallel_group_size = 2, expert_parallel_replica_group_size = 2)
    (nrank_0, nid_0), (nrank_1, nid_1) = dgrid.map_expert_id_global_to_local(64, 42)
    assert nrank_0 == 1 and nid_0 == 10 and nrank_1 == 3 and nid_1 == 10

    grank , replica_id = dgrid.map_expert_id_local_to_global(64, 14)
    if rank == 0:
        assert grank == 14 and replica_id == 0
    elif rank == 1:
        assert grank == 46 and replica_id == 0
    elif rank == 2:
        assert grank == 14 and replica_id == 1
    elif rank == 3:
        assert grank == 46 and replica_id == 1

def test_expert_mapping_4(device = rank):
    if dist.get_world_size() != 4:
        return

    dgrid = DistributionGrid(expert_parallel_group_size = 4)

    grank = dgrid.map_expert_id_local_to_global(64, 14)
    if rank == 0:
        assert grank == 14
    elif rank == 1:
        assert grank == 30
    elif rank == 2:
        assert grank == 46
    elif rank == 3:
        assert grank == 62

    dgrid.exchange_expert_location(30, 30)
    dgrid.exchange_expert_location(30, 46)    

    nrank, nid = dgrid.map_expert_id_global_to_local(64, 30)
    assert nrank == 2 and nid == 14

    grank2 = dgrid.map_expert_id_local_to_global(64, 14)
    if rank == 0:
        assert grank2 == 14
    elif rank == 1:
        assert grank2 == 46
    elif rank == 2:
        assert grank2 == 30
    elif rank == 3:
        assert grank2 == 62

    dgrid.set_expert_relocation_map(dgrid.get_expert_relocation_map())
    dgrid.remove_expert_relocation(30)

def test_rank_schedule(device = rank):
    if dist.get_world_size() != 4:
        return

    dgrid = DistributionGrid(expert_parallel_group_size = 2, expert_parallel_replica_group_size = 2)
    # EP 0 -> [0,1]
    # EP 1 -> [2,3]
    # ER 0 -> [0,2]
    # ER 1 -> [1,3]
    if (rank % 2) == 0:
        assert dgrid.get_expert_replica_src_rank() == 0
        assert dgrid.get_expert_parallel_replica_src_rank() == 0
    else:
        assert dgrid.get_expert_replica_src_rank() == 1
    if rank < 2:
        assert dgrid.get_expert_parallel_replica_rank() == 0
    else:
        assert dgrid.get_expert_parallel_replica_rank() == 1

    options = {"rank_schedule": "row_major"}
    dgridr = DistributionGrid(expert_parallel_group_size = 2, expert_parallel_replica_group_size = 2,
                             options = options)
    # EP 0 -> [0,1]
    # EP 1 -> [2,3]
    # ER 0 -> [0,2]
    # ER 1 -> [1,3]
    if (rank % 2) == 0:
        assert dgridr.get_expert_replica_src_rank() == 0
    else:
        assert dgridr.get_expert_replica_src_rank() == 1
    if rank < 2:
        assert dgridr.get_expert_parallel_replica_rank() == 0
    else:
        assert dgridr.get_expert_parallel_replica_rank() == 1

    options = {"rank_schedule": "column_major"}
    dgridc = DistributionGrid(expert_parallel_group_size = 2, expert_parallel_replica_group_size = 2,
                              options = options)
    # EP 0 -> [0,2]
    # EP 1 -> [1,3]
    # ER 0 -> [0,1]
    # ER 1 -> [2,3]
    if rank < 2:
        assert dgridc.get_expert_replica_src_rank() == 0
    else:
        assert dgridc.get_expert_replica_src_rank() == 2
    if (rank % 2) == 0:
        assert dgridc.get_expert_parallel_replica_rank() == 0
    else:
        assert dgridc.get_expert_parallel_replica_rank() == 1

    options = {"rank_schedule": "row_major", "is_replica_in_same_node": True}
    dgridr = DistributionGrid(expert_parallel_group_size = 2, expert_parallel_replica_group_size = 2,
                              options = options)
    # EP 0 -> [0,2]
    # EP 1 -> [1,3]
    # ER 0 -> [0,1]
    # ER 1 -> [2,3]
    if rank < 2:
        assert dgridr.get_expert_replica_src_rank() == 0
    else:
        assert dgridr.get_expert_replica_src_rank() == 2
    if (rank % 2) == 0:
        assert dgridr.get_expert_parallel_replica_rank() == 0
    else:
        assert dgridr.get_expert_parallel_replica_rank() == 1
