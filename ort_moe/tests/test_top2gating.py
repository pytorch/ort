# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
#importing tests from fairscale and modifying based on it.
#https://github.com/facebookresearch/fairscale/blob/master/tests/nn/moe/test_top2gating.py
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# Implementation of Top2Gating described in https://arxiv.org/pdf/2006.16668.pdf
# Code is inspired by Top2GatingOnLogits from lingvo:
#   https://github.com/tensorflow/lingvo/blob/21b8106c5f1d30a196c98eedc441d4fd70833b11/lingvo/core/moe_layers.py#L477
import pytest
import torch
import math
import os

from mpi4py import MPI
import torch.distributed as dist
from ort_moe.topKgate import TopKGate, top2gating, top1gating, fast_one_hot, balance_ratio_to_dict
from ort_moe.loss_functions import loss_functions
from ort_moe.gate_logs import gate_logs
from ort_moe.grids import DistributionGrid
from ort_moe.moe_config import moe_config

#Uncomment this if there is no cuda
#skip_if_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")

dgrid = DistributionGrid()
BACKEND = dist.Backend.NCCL
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29502"  # torch 1.5 compatibility
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if not dist.is_initialized():
    dist.init_process_group(backend=BACKEND, rank=rank, world_size=size)

def test_create():
    gate = TopKGate(4, 8, dgrid=dgrid, k = 1)
    p_total = sum(p.numel() for p in gate.parameters())
    assert p_total == 4 * 8

#@skip_if_no_cuda
def test_create_cuda():
    gate = TopKGate(4, 8, dgrid=dgrid).cuda()
    p_total = sum(p.numel() for p in gate.parameters())
    assert p_total == 4 * 8

def do_test_forward(device='cpu', topk = 1, fp16_mode = False):
    torch.manual_seed(3)
    num_tokens = 18
    num_experts = 6
    model_dim = 4

    input = torch.randn(num_tokens, model_dim).to(device)
    options = {}
    options["fp16_mode"] = fp16_mode
    gate = TopKGate(model_dim, num_experts, dgrid=dgrid, k = topk, options = options).to(device)
    capacity_fp = max(min(num_tokens, (topk * num_tokens / num_experts)), 4)
    capacity = math.ceil(capacity_fp)

    combine_weights, dispatch_mask, capacity_fp = gate(input)
    assert dispatch_mask.shape == torch.Size([num_experts, capacity]) #EC
    assert combine_weights.shape == torch.Size([topk, num_tokens]) #KS
    assert torch.all(dispatch_mask < num_tokens * topk)
    #below is copied from fairscale https://github.com/facebookresearch/fairscale/blob/master/tests/nn/moe/test_top2gating.py
    if topk == 2:
        weights_sum = torch.sum(combine_weights).item()
        assert pytest.approx(gate.loss.item()) == 0.010920888744294643
        #assert round(weights_sum) == pytest.approx(weights_sum)
        # For this seed, for top-2, we get #tokens slots filled.
        assert weights_sum == pytest.approx(num_tokens)

def test_forward_cpu():
    do_test_forward("cpu", topk=2)
    do_test_forward("cpu", topk=1)
    do_test_forward("cpu", topk=2, fp16_mode = True)
    do_test_forward("cpu", topk=1, fp16_mode = True)

#@skip_if_no_cuda
def test_forward_cuda():
    do_test_forward(rank, topk=2)
    do_test_forward("cpu", topk=1)


#verify that top gate is allocated capacity
def do_test_topks(k):
    num_tokens = 8
    num_experts = 4
    logits = torch.randn(num_tokens, num_experts)
    if k == 2:
        loss, _, _, dispatch_mask, capacity_fp = top2gating(logits, capacity_factor=1)
    else:
        loss, _, _, dispatch_mask, capacity_fp = top1gating(logits, capacity_factor=1)
    top1s = torch.argmax(logits, dim=1)
    capacity = k * num_tokens // num_experts
    ce = [0] * num_tokens
    for i, s in enumerate(top1s):
        e = s.item()
        loc = ce[e]
        ce[e] = loc+1
        if ce[e] < capacity:
            assert (dispatch_mask[e][loc] == i or dispatch_mask[e][loc] == i + num_tokens)

def test_topks():
    do_test_topks(1)
    do_test_topks(2)

def test_fast_one_hot():
    num_class = 4
    input = torch.rand(3, num_class)
    indices = torch.argmax(input, dim = 1)
    one_hot_ref = torch.nn.functional.one_hot(indices, num_class)
    one_hot_moe = fast_one_hot(indices, num_class)
    assert torch.equal(one_hot_ref, one_hot_moe)

# Test top1gating of only dispatching the nonpadding tokens.
# Same with do_test_topks(1) except that 1) nonpadding mask is passed to the top1gating() and
# 2) only check the dispatch mask of nonpadding tokens.
def do_test_nonpadding_top1():
    num_tokens = 8
    num_experts = 4
    num_nonpadding = 5
    logits = torch.randn(num_tokens, num_experts)
    nonpadding = torch.zeros(num_tokens).to(torch.int64).scatter_(0, torch.arange(0, num_nonpadding), 1)
    br={'z_loss': 0.1}
    _, _, _, dispatch_mask, _ = top1gating(logits, capacity_factor=1, balance_ratio=br, nonpadding=nonpadding, straight_through=True)
    top1s = torch.argmax(logits, dim=1)
    capacity = num_tokens // num_experts
    ce = [0] * num_tokens
    for i, s in enumerate(top1s):
        e = s.item()
        loc = ce[e]
        ce[e] = loc+1
        if ce[e] < capacity and i < num_nonpadding:
            # only nonpadding tokens within capacity have dispatch mask
            assert dispatch_mask[e][loc] == i or dispatch_mask[e][loc] == i + num_tokens
def do_test_nonpadding_top2():
    num_tokens = 8
    num_experts = 4
    num_nonpadding = 5
    logits = torch.randn(num_tokens, num_experts)
    nonpadding = torch.zeros(num_tokens).to(torch.int64).scatter_(0, torch.arange(0, num_nonpadding), 1)
    _, _, _, dispatch_mask, _ = top2gating(logits, capacity_factor=1, nonpadding=nonpadding, straight_through=True)
    top1s = torch.argmax(logits, dim=1)
    logits_except1 = logits.masked_fill(torch.nn.functional.one_hot(top1s, num_classes=num_experts).bool(), float("-inf"))
    top2s = torch.argmax(logits_except1, dim=1)
    top1s = torch.cat([top1s, top2s])
    capacity = 2 * num_tokens // num_experts
    ce = [0] * num_tokens
    for i, s in enumerate(top1s):
        e = s.item()
        loc = ce[e]
        ce[e] = loc+1
        if ce[e] < capacity and i < num_nonpadding:
            # only nonpadding tokens within capacity have dispatch mask
            assert dispatch_mask[e][loc] == i or dispatch_mask[e][loc] == i + num_tokens
def test_nonpadding():
    do_test_nonpadding_top1()
    do_test_nonpadding_top2()

def do_test_loss(k, balance_ratio=None, gate_log_req=None, device=None, options=None):
    # verify that refactored loss computation can get the same result with old version code
    num_tokens = 8
    num_experts = 4
    logits = torch.randn(num_tokens, num_experts, device=device)

    if k == 2:

        if balance_ratio is None and gate_log_req is None:
            loss, _, _, _, _ = top2gating(logits, capacity_factor=1, logits_gumbel=1, options=options)
        elif gate_log_req is None:
            loss, _, _, _, _ = top2gating(logits, capacity_factor=1, logits_gumbel=1, balance_ratio=balance_ratio_to_dict(balance_ratio), straight_through=True, straight_through_temperature=1.0, options=options)
            loss1, _, _, _, _ = top2gating(logits, capacity_factor=1, logits_gumbel=1, balance_ratio=balance_ratio_to_dict(balance_ratio), straight_through=True, straight_through_temperature=1.0-1e-9, options=options)
            loss2, _, _, _, _ = top2gating(logits, capacity_factor=1, logits_gumbel=0, balance_ratio=balance_ratio_to_dict(balance_ratio), straight_through=True, straight_through_temperature=1.0, options=options)
            loss3, _, _, _, _ = top2gating(logits, capacity_factor=1, logits_gumbel=0, balance_ratio=balance_ratio_to_dict(balance_ratio), straight_through=True, straight_through_temperature=1.0-1e-9, options=options)
            assert loss == loss1 == loss2 == loss3
        elif balance_ratio is None:
            loss, _, _, _, _ = top2gating(logits, capacity_factor=1, logits_gumbel=1, gate_log_req=gate_log_req, options=options)
        else:
            loss, _, _, _, _ = top2gating(logits, capacity_factor=1, logits_gumbel=1, balance_ratio=balance_ratio_to_dict(balance_ratio), gate_log_req=gate_log_req, options=options)

        if balance_ratio is None:
            balance_ratio = 0.01
        balance_ratio = balance_ratio_to_dict(balance_ratio)

    else:
        if balance_ratio is not None:
            balance_ratio = balance_ratio_to_dict(balance_ratio)

        if balance_ratio is None and gate_log_req is None:
            loss, gate_log, _, _, _ = top1gating(logits, capacity_factor=1, options=options)
        elif gate_log_req is None:
            loss, gate_log, _, _, _ = top1gating(logits, capacity_factor=1, balance_ratio=balance_ratio, straight_through=True, straight_through_temperature=1.0, logits_gumbel=0, options=options)
            loss1, _, _, _, _ = top1gating(logits, capacity_factor=1, balance_ratio=balance_ratio, straight_through=True, straight_through_temperature=1.0-1e-9, logits_gumbel=0, options=options)
            loss2, _, _, _, _ = top1gating(logits, capacity_factor=1, balance_ratio=balance_ratio, straight_through=True, straight_through_temperature=1.0, logits_gumbel=1e-9, options=options)
            loss3, _, _, _, _ = top1gating(logits, capacity_factor=1, balance_ratio=balance_ratio, straight_through=True, straight_through_temperature=1.0-1e-9, logits_gumbel=1e-9, options=options)
            assert loss == loss1 == loss2 == loss3
        elif balance_ratio is None:
            loss, gate_log, _, _, _ = top1gating(logits, capacity_factor=1, gate_log_req=gate_log_req, options=options)
        else:
            loss, gate_log, _, _, _ = top1gating(logits, capacity_factor=1, balance_ratio=balance_ratio, gate_log_req=gate_log_req, options=options)

def test_loss():
    balance_ratio={'load_balance': 0.1, 'sparsity_l1': 0.1, 'mean_importance': 0.1, 'z_loss': 0.1, 'ideal_load_balance': 1e-9}
    gate_log_req={'gate_entropy': True, 'gate_probability': True, 'gate_routed': True, 'expert_fraction': True, 'expert_routed_fraction': True}

    options = {}
    options["enable_tutel_cumsum"] = True

    do_test_loss(2)
    do_test_loss(2, device=rank, options=moe_config(options))
    do_test_loss(2, balance_ratio=0.01)
    do_test_loss(2, gate_log_req=gate_log_req)
    do_test_loss(2, balance_ratio=0.01, gate_log_req=gate_log_req)

    do_test_loss(1)
    do_test_loss(1, device=rank, options=moe_config(options))
    do_test_loss(1, balance_ratio=balance_ratio)
    do_test_loss(1, gate_log_req=gate_log_req)
    do_test_loss(1, balance_ratio=balance_ratio, gate_log_req=gate_log_req)

# Test token drop type for 'cut' and 'routing_weight'
def do_test_token_drop_top1(token_drop_type):
    num_tokens = 9
    num_experts = 3
    logits = torch.randn(num_tokens, num_experts)
    _, _, _, dispatch_mask, _ = top1gating(logits, capacity_factor=1, token_drop_type=token_drop_type, straight_through=True)
    top1s = torch.argmax(logits, dim=1)
    capacity = max(min(num_tokens, (num_tokens // num_experts)), 4)
    ce = [0] * num_tokens
    if token_drop_type == 'cut':
        for i, s in enumerate(top1s):
            e = s.item()
            loc = ce[e]
            ce[e] = loc+1
            if ce[e] < capacity:
                assert dispatch_mask[e][loc] == i or dispatch_mask[e][loc] == i + num_tokens
    elif token_drop_type == 'routing_weight':
        mask1 = torch.nn.functional.one_hot(top1s, num_classes=num_experts)
        gates = torch.nn.functional.softmax(logits, dim=1)
        _, cap_idx = torch.topk(mask1*gates, k=min(capacity, gates.shape[0]), dim=0)
        for i, s in enumerate(top1s):
            e = s.item()
            if mask1[i, e] and i in cap_idx[:, e]:
                loc = ce[e]
                ce[e] = loc+1
                assert dispatch_mask[e][loc] == i or dispatch_mask[e][loc] == i + num_tokens
def do_test_token_drop_top2(token_drop_type):
    num_tokens = 9
    num_experts = 3
    logits = torch.randn(num_tokens, num_experts)
    _, _, _, dispatch_mask, _ = top2gating(logits, capacity_factor=1, token_drop_type=token_drop_type, straight_through=True)
    top1s = torch.argmax(logits, dim=1)
    logits_except1 = logits.masked_fill(torch.nn.functional.one_hot(top1s, num_classes=num_experts).bool(), float("-inf"))
    top2s = torch.argmax(logits_except1, dim=1)
    top1s = torch.cat([top1s, top2s])
    capacity = max(min(num_tokens, (2 * num_tokens // num_experts)), 4)
    ce = [0] * num_tokens
    if token_drop_type == 'cut':
        for i, s in enumerate(top1s):
            e = s.item()
            loc = ce[e]
            ce[e] = loc+1
            if ce[e] < capacity:
                assert dispatch_mask[e][loc] == i or dispatch_mask[e][loc] == i + num_tokens
    elif token_drop_type == 'routing_weight':
        mask1 = torch.nn.functional.one_hot(top1s, num_classes=num_experts)
        gates1 = torch.nn.functional.softmax(logits, dim=1)
        gates1 = torch.cat([gates1, gates1])
        _, cap_idx = torch.topk(mask1*gates1, k=min(capacity, gates1.shape[0]), dim=0)
        for i, s in enumerate(top1s):
            e = s.item()
            if mask1[i, e] and i in cap_idx[:, e]:
                loc = ce[e]
                ce[e] = loc+1
                assert dispatch_mask[e][loc] == i or dispatch_mask[e][loc] == i + num_tokens
def test_token_drop():
    do_test_token_drop_top1(token_drop_type='cut')
    do_test_token_drop_top1(token_drop_type='routing_weight')

    do_test_token_drop_top2(token_drop_type='cut')
    do_test_token_drop_top2(token_drop_type='routing_weight')
def test_tutel_cumsum():
    matrix = torch.randint(0, 100, (10000, 100), device=rank)
    from tutel.jit_kernels.gating import fast_cumsum_sub_one
    cumsum_tutel = fast_cumsum_sub_one(matrix, dim=0) + 1
    cumsum_torch = torch.cumsum(matrix, dim=0)
    assert cumsum_tutel.equal(cumsum_torch), "Result of tutel's cumsum is not correct"

# Test dynamic capacity in top1gating when the max_token_usage based capacity is in effect.
def test_top1gating_dynamic_capacity():
    dgrid = DistributionGrid(expert_parallel_group_size = 4)
    rank_list = list(range(0, dist.get_world_size()))
    ranks_count = 4
    logits = torch.eye(128, dtype=torch.float32)
    logits = logits.repeat(3, 1)
    options = {}
    options["enable_dynamic_capacity"] = True
    _, _, _, _, capacity_fp = top1gating(logits, capacity_factor=100.0, options=options)
    assert capacity_fp == 32.0
    logits = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    logits = logits.repeat(3, 1)
    _, _, _, _, capacity_fp = top1gating(logits, capacity_factor=5.0, options=options)
    assert capacity_fp == 4.0
