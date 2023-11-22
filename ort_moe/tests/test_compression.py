# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from mpi4py import MPI
import torch.distributed as dist
import torch
import pytest
import os

from ort_moe.collectives import compressed_all_to_all

assert torch.cuda.is_available()

BACKEND = dist.Backend.NCCL
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29501"  # torch 1.5 compatibility
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if not dist.is_initialized():
    dist.init_process_group(backend=BACKEND, rank=rank, world_size=size)

def compare_tensor_lists(l1, l2):
    for a, b in zip(l1, l2):
        return torch.equal(a, b)

def test_fake(device=rank):
    return

def disable_test_alltoall_compression(device = rank):
    # generate input
    a2a_input = torch.empty(4*2*8, dtype=torch.float16, device=torch.device(rank))
    a2a_input = torch.reshape(a2a_input,[4,2,8])
    a2a_input[0] = a2a_input[0].fill_(rank*2)
    a2a_input[1] = a2a_input[1].fill_(rank*3)
    a2a_input[2] = a2a_input[2].fill_(rank*4)
    a2a_input[3] = a2a_input[3].fill_(rank*5)

    # original all2all
    orig_a2a_output = torch.empty(a2a_input.size(), dtype=a2a_input.dtype, device=a2a_input.device)
    torch.distributed.all_to_all_single(orig_a2a_output, a2a_input)

    # manual all2all
    mc_a2a_output = torch.empty(a2a_input.size(), dtype=a2a_input.dtype, device=a2a_input.device)
    compressed_all_to_all(mc_a2a_output, a2a_input)

    assert compare_tensor_lists([orig_a2a_output], [mc_a2a_output])
