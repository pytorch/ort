# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from _pytest.mark import param

from mpi4py import MPI
import torch.distributed as dist
import torch
import pytest
import os
import tempfile
import math
import random

from torch import nn

from ort_moe.custom_ops import om_einsum
from ort_moe.experts import FFNExpert, MergedFFNExpert
from ort_moe.moe import MixtureOfExpertsFunc, MixtureOfExperts, AllToAll, MixtureOfExpertsES
from ort_moe.utils import broadcast_parameters, moe_module_all_reduce, apex_amp_scale_check_overflow_override, is_moe_parameter
from ort_moe.utils import get_expert_parameters_state_dict, get_non_expert_parameters_state_dict
from ort_moe.utils import get_expert_parameters_list, get_non_expert_parameters_list
from ort_moe.utils import get_state_dict_for_local_expert_idx
from ort_moe.utils import get_state_dict_partitions_for_saving, get_state_dict_partition_names_for_loading
from ort_moe.utils import translate_state_dict_key_global_to_local, translate_state_dict_key_local_to_global
from ort_moe.utils import translate_state_dict_global_to_local, translate_state_dict_local_to_global
from ort_moe.utils import get_moe_loss, fsdp_wrap
from ort_moe.topKgate import TopKGate
from ort_moe.layers import LanguageExpertMoEEncoderLayer, LanguageExpertMoEDecoderLayer
from ort_moe.layers import TransformerMoEEncoderLayer, TransformerMoEDecoderLayer
from ort_moe.grids import DistributionGrid
from ort_moe.loss_functions import loss_functions
from ort_moe.gate_logs import gate_logs
from apex import amp as apex_amp

# if USE_ORT env is set then run tests with ORTModule
use_ort = os.getenv("USE_ORT", None)
if use_ort:
    import ort_moe.custom_ops
    from onnxruntime.training.ortmodule import DebugOptions, LogLevel
    from onnxruntime.training import ortmodule
    from onnxruntime.training.ortmodule._custom_autograd_function import enable_custom_autograd_support
    from torch_ort import ORTModule
    ort_moe.custom_ops.USE_EINSUM = False
    enable_custom_autograd_support()
    debug_options = DebugOptions(save_onnx=True, onnx_prefix='moe', log_level=LogLevel.INFO)
    ortmodule.ONNX_OPSET_VERSION=13

assert torch.cuda.is_available()

BACKEND = dist.Backend.NCCL
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29501"  # torch 1.5 compatibility
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if not dist.is_initialized():
    dist.init_process_group(backend=BACKEND, rank=rank, world_size=size)

@pytest.mark.with_ort
def testFFNExpert_init():
    dgrid = DistributionGrid()
    model_dim = 4
    ff_dim = 6
    ffn = FFNExpert(model_dim, ff_dim, dgrid=dgrid)
    if use_ort:
        ffn = ORTModule(ffn)
    assert sum(p.numel() for p in ffn.parameters()) == 2 * model_dim * ff_dim

@pytest.mark.with_ort
def testFFNExpert_forward():
    dgrid = DistributionGrid()
    model_dim = 8  # = ff_dim
    input = torch.randn(12, model_dim)
    ffn = FFNExpert(model_dim, model_dim, dgrid)
    if use_ort:
        ffn = ORTModule(ffn)
    # use identify matrix
    ffn.linear1.weight = torch.nn.Parameter(torch.eye(model_dim))
    ffn.linear2.weight = torch.nn.Parameter(torch.eye(model_dim))
    output = ffn(input)
    if use_ort:
        loss = output[0].sum()
        loss.backward()
    assert input.shape == output.shape
    assert torch.allclose(input, torch.where(output > 0, output, input))

@pytest.mark.with_ort
def test_MixtureOfExperts():
    dgrid = DistributionGrid(expert_parallel_group_size=dist.get_world_size(dist.group.WORLD))
    model_dim = 8
    ff_dim = 12

    num_local_experts = 4
    num_experts = dist.get_world_size(dist.group.WORLD) * num_local_experts

    gating_fn = TopKGate(model_dim, num_experts, k=1, dgrid=dgrid)  # Top1
    experts = torch.nn.ModuleList()
    for i in range(num_local_experts):
        experts.append(FFNExpert(model_dim, ff_dim, dgrid=dgrid))

    me = MixtureOfExpertsFunc(gating_fn, experts, distribution_grid = dgrid)
    if use_ort:
        me = ORTModule(me)

    assert len(get_expert_parameters_list(me)) + len(get_non_expert_parameters_list(me)) == \
       len(list(me.parameters()))
    for p in get_expert_parameters_list(me):
       assert is_moe_parameter(p)
    for p in get_non_expert_parameters_list(me):
       assert not is_moe_parameter(p)

     #Check the state dicts for experts and non-experts
    for k in me.state_dict().keys():
        if "moe_experts" in k:
            assert k in (get_expert_parameters_state_dict(me))
        else:
            assert k in (get_non_expert_parameters_state_dict(me))

    # test get_state_dict_for_local_expert_idx
    for i in range(num_local_experts):
        expert_state_dict = get_state_dict_for_local_expert_idx(me, i)
        for k in me.state_dict().keys():
            keyword = f"moe_experts.{i}"
            if keyword in k:
                assert k in expert_state_dict
            else:
                assert k not in expert_state_dict

    # test state_dict translation
    for k in get_non_expert_parameters_state_dict(me).keys():
        assert translate_state_dict_key_local_to_global(k, dgrid, num_experts) == k
        assert translate_state_dict_key_global_to_local(k, dgrid, num_experts) == k
    for i in range(num_local_experts):
        keyword = f"moe_experts.{i}"
        global_index = dist.get_rank() * num_local_experts + i
        replace_to = f"moe_experts.{global_index}"
        for local_key in me.state_dict().keys():
            if keyword in local_key:
                global_key = local_key.replace(keyword, replace_to)
                assert translate_state_dict_key_local_to_global(local_key, dgrid, num_experts) == global_key
                assert translate_state_dict_key_global_to_local(global_key, dgrid, num_experts) == local_key
    global_state_dict = translate_state_dict_local_to_global(me.state_dict(), dgrid, num_experts)
    local_state_dict = translate_state_dict_global_to_local(global_state_dict, dgrid, num_experts)
    assert list(local_state_dict.keys()) == list(me.state_dict().keys())

    # test get_state_dict_partitions_for_saving
    partitions = get_state_dict_partitions_for_saving(me, dgrid, num_experts)
    if dist.get_rank() == 0:
        assert len(partitions) == num_local_experts + 1
        assert "skeleton" in partitions.keys()
        assert list(partitions["skeleton"].keys()) == list(get_non_expert_parameters_state_dict(me).keys())
    else:
        assert len(partitions) == num_local_experts
    for i in range(num_local_experts):
        global_index = dist.get_rank() * num_local_experts + i
        key = f"expert{global_index}"
        assert key in partitions.keys()
        local_state_dict = translate_state_dict_global_to_local(partitions[key], dgrid, num_experts)
        assert list(local_state_dict.keys()) == list(get_state_dict_for_local_expert_idx(me, i).keys())

    # test get_state_dict_partition_names_for_loading
    partition_names = get_state_dict_partition_names_for_loading(me, dgrid, num_experts)
    assert len(partition_names) == num_local_experts + 1
    assert "skeleton" in partition_names
    for i in range(num_local_experts):
        global_index = dist.get_rank() * num_local_experts + i
        key = f"expert{global_index}"
        assert key in partition_names

    #number of parameters match
    assert sum(p.numel() for p in me.parameters()) == \
        num_local_experts * 2 * ff_dim * model_dim  \
        + model_dim * num_experts

def test_save_load_expert_replicas():
    if not (dist.get_world_size() >= 4 and dist.get_world_size() % 2 == 0):
        return 0

    dgrid = DistributionGrid(expert_parallel_group_size=dist.get_world_size() // 2,
                             expert_parallel_replica_group_size=2)
    model_dim = 8
    ff_dim = 12

    num_local_experts = 2
    num_experts = dist.get_world_size() // 2 * num_local_experts

    gating_fn = TopKGate(model_dim, num_experts, k=1, dgrid=dgrid)  # Top1
    experts = torch.nn.ModuleList()
    for i in range(num_local_experts):
        experts.append(FFNExpert(model_dim, ff_dim, dgrid=dgrid))

    model = MixtureOfExpertsFunc(gating_fn, experts, distribution_grid=dgrid)

    # test state_dict translation
    for k in get_non_expert_parameters_state_dict(model).keys():
        assert translate_state_dict_key_local_to_global(k, dgrid, num_experts) == k
        assert translate_state_dict_key_global_to_local(k, dgrid, num_experts) == k
    for i in range(num_local_experts):
        keyword = f"moe_experts.{i}"
        global_index = (dist.get_rank() * num_local_experts + i) % num_experts
        replace_to = f"moe_experts.{global_index}"
        for local_key in model.state_dict().keys():
            if keyword in local_key:
                global_key = local_key.replace(keyword, replace_to)
                assert translate_state_dict_key_local_to_global(local_key, dgrid, num_experts) == global_key
                assert translate_state_dict_key_global_to_local(global_key, dgrid, num_experts) == local_key
    global_state_dict = translate_state_dict_local_to_global(model.state_dict(), dgrid, num_experts)
    local_state_dict = translate_state_dict_global_to_local(global_state_dict, dgrid, num_experts)
    assert list(local_state_dict.keys()) == list(model.state_dict().keys())

    # test get_state_dict_partitions_for_saving
    partitions = get_state_dict_partitions_for_saving(model, dgrid, num_experts)
    if dist.get_rank() == 0:
        assert len(partitions) == num_local_experts + 1
        assert "skeleton" in partitions.keys()
        assert list(partitions["skeleton"].keys()) == list(get_non_expert_parameters_state_dict(model).keys())

    if dgrid.get_expert_replica_rank() == 0:
        if dist.get_rank() != 0:
            assert len(partitions) == num_local_experts
        for i in range(num_local_experts):
            global_index = (dist.get_rank() * num_local_experts + i) % num_experts
            key = f"expert{global_index}"
            assert key in partitions.keys()
            local_state_dict = translate_state_dict_global_to_local(partitions[key], dgrid, num_experts)
            assert list(local_state_dict.keys()) == list(get_state_dict_for_local_expert_idx(model, i).keys())
    else:
        assert len(partitions) == 0

    # test get_state_dict_partition_names_for_loading
    partition_names = get_state_dict_partition_names_for_loading(model, dgrid, num_experts)
    assert len(partition_names) == num_local_experts + 1
    assert "skeleton" in partition_names
    for i in range(num_local_experts):
        global_index = (dist.get_rank() * num_local_experts + i) % num_experts
        key = f"expert{global_index}"
        assert key in partition_names

@pytest.mark.with_ort
def test_MixtureOfExperts_single_forward(device=rank):
    model_dim = 8
    num_local_experts = 1
    num_experts = dist.get_world_size(dist.group.WORLD) * num_local_experts
    input = torch.rand(4, 16, model_dim).to(device)
    dgrid = DistributionGrid(expert_parallel_group_size = dist.get_world_size())
    gate = TopKGate(model_dim, num_experts, k=2, dgrid=dgrid)
    experts = torch.nn.ModuleList()
    for i in range(num_local_experts):
        expert = torch.nn.Linear(model_dim, model_dim, bias=False)
        # use identify matrix
        expert.weight = torch.nn.Parameter(torch.eye(model_dim))
        experts.append(expert)
    moe = MixtureOfExpertsFunc(gate, experts, distribution_grid = dgrid).to(device)
    if use_ort:
        moe = ORTModule(moe)
    output = moe(input)
    if use_ort:
        loss = output[0].sum()
        loss.backward()
    assert output.shape == input.shape
    # Re-assembled output should match input due to identity expert.
    assert torch.allclose(input, torch.where(output > 0, output, input))

# launch with mpirun -n x python -m pytest --with-mpi xxx.py


@pytest.mark.mpi
@pytest.mark.with_ort
def test_MixtureOfExperts_multi_forward(device=rank):
    num_local_experts = 4
    model_dim = 4
    num_experts = dist.get_world_size(dist.group.WORLD) * num_local_experts
    input = torch.rand(4 * num_local_experts, 16, model_dim).to(device)
    dgrid = DistributionGrid(expert_parallel_group_size = dist.get_world_size())
    gate = TopKGate(model_dim, num_experts, k=2, dgrid=dgrid)
    experts = torch.nn.ModuleList()
    for i in range(num_local_experts):
        expert = torch.nn.Linear(model_dim, model_dim, bias=False)
        # use identify matrix
        expert.weight = torch.nn.Parameter(torch.eye(model_dim))
        experts.append(expert)
    moe = MixtureOfExpertsFunc(gate, experts, distribution_grid = dgrid).to(device)
    if use_ort:
        moe = ORTModule(moe)
    output = moe(input)
    if use_ort:
        loss = output[0].sum()
        loss.backward()
    assert output.shape == input.shape
    # The following assertions only holds for k = 2, because combined_weights
    # are normalized in top2 but not top1
    # Except for zeros, re-assembled output should match input due to identity expert.
    assert torch.allclose(input, torch.where(output > 0, output, input))


@pytest.mark.with_ort
def test_expert_count():
    dgrid = DistributionGrid(expert_parallel_group_size = dist.get_world_size())
    encoder = TransformerMoEEncoderLayer(256, 4, nexperts=8, distribution_grid = dgrid)
    if use_ort:
        encoder = ORTModule(encoder)
    assert encoder.n_local_experts == 8 / size


@pytest.mark.with_ort
def test_expert_moe_encoder():
    dgrid = DistributionGrid(expert_parallel_group_size = dist.get_world_size())
    d_model = 512
    nhead = 8
    dim_feedforward = 256
    num_local_experts = 4
    nexperts = dist.get_world_size(dist.group.WORLD) * num_local_experts
    gate = TopKGate(d_model, nexperts, k=1, dgrid=dgrid)  # top-1
    encoder = TransformerMoEEncoderLayer(d_model, nhead, dim_feedforward=dim_feedforward,
                                         nexperts=nexperts, gate=gate, distribution_grid = dgrid)
    if use_ort:
        encoder = ORTModule(encoder)
    total_parameters = sum(p.numel() for p in encoder.parameters())
    expected_parameters = d_model * nexperts  # number of parameters in gate
    expected_parameters += num_local_experts * 2 * d_model * \
        dim_feedforward  # number of parameters in Experts
    # number of attention and the W_o
    expected_parameters += 4 * (d_model * d_model + d_model)
    expected_parameters += 4 * d_model  # number of layernorms
    assert total_parameters == expected_parameters

# TODO:
# def test_expert_moe_encoder_forward():


@pytest.mark.with_ort
def test_expert_moe_decoder():
    d_model = 512
    nhead = 8
    dim_feedforward = 256
    num_local_experts = 4
    nexperts = dist.get_world_size(dist.group.WORLD) * num_local_experts
    dgrid = DistributionGrid(expert_parallel_group_size = dist.get_world_size())
    gate = TopKGate(d_model, nexperts, k=1, dgrid=dgrid)  # top-1
    decoder = TransformerMoEDecoderLayer(d_model, nhead, dim_feedforward=dim_feedforward,
                                         nexperts=nexperts, gate=gate, distribution_grid = dgrid)
    if use_ort:
        decoder = ORTModule(decoder)

    total_paramters = sum(p.numel() for p in decoder.parameters())
    expected_parameters = d_model * nexperts  # number of parameters in gate
    expected_parameters += num_local_experts * 2 * d_model * \
        dim_feedforward  # number of parameters in Experts
    # number of attention and the W_o
    expected_parameters += 2 * 4 * (d_model * d_model + d_model)
    expected_parameters += 6 * d_model  # number of layernorms
    assert total_paramters == expected_parameters

# TODO:
# def test_expert_moe_decoder_forward():


# Below is imported from fairscale
# Test Gate which round-robin routes tokens to experts


class RoundRobinGate(torch.nn.Module):
    def __init__(self, model_dim, num_experts, dgrid):
        super().__init__()
        self.model_dim = model_dim
        self.num_experts = num_experts
        self.loss = None
        self.gate_log = None

    def forward(self, input, *args, **kwargs):
        s = int(input.shape[0])
        assert s % self.num_experts == 0
        capacity_fp = 2 * s / self.num_experts
        capacity = int(math.ceil(capacity_fp))
        assert self.num_experts * capacity % s == 0
        dispatch_mask = torch.zeros(self.num_experts, capacity, dtype = torch.long, device = input.device)
        gates = torch.ones(2, s, dtype = input.dtype, device = input.device)
        self.k = self.num_experts * capacity // s
        assert self.k <= 2
        gates *= 1.0/self.k
        for token in range(self.k * input.shape[0]):
            if token > self.num_experts * capacity:
                break
            x_idx = (token  % self.num_experts)
            y_idx = (token // self.num_experts)
            dispatch_mask[x_idx][y_idx] = token

        self.loss = torch.tensor(0.0)
        self.gate_log = {}
        return gates, dispatch_mask, capacity_fp


@pytest.mark.mpi
@pytest.mark.with_ort
def test_forward_routing_multi(device=rank):
    dgrid = DistributionGrid(expert_parallel_group_size = dist.get_world_size())
    model_dim = 8
    num_local_experts = 4
    num_experts = dist.get_world_size(dist.group.WORLD) * num_local_experts
    input = torch.ones(4 * num_local_experts, 32, model_dim).to(device)
    gate = RoundRobinGate(model_dim, num_experts, dgrid)
    experts = torch.nn.ModuleList()
    for i in range(num_local_experts):
        expert = torch.nn.Linear(model_dim, model_dim, bias=False)
        # Use scaling matrix (each rank has a different scale)
        scale = dist.get_rank() * num_local_experts + i + 1
        expert.weight = torch.nn.Parameter(torch.eye(model_dim) * scale)
        experts.append(expert)
    moe = MixtureOfExpertsFunc(gate, experts, distribution_grid = dgrid).to(device)
    if use_ort:
        moe = ORTModule(moe)
    output = moe(input)
    if use_ort:
        loss = output[0].sum()
        loss.backward()
    assert output.shape == input.shape
    # Verify that each token was sent to the correct expert by checking its scale.
    t = input.shape[1]
    for i in range(t):
        expert = i % num_experts
        assert torch.allclose(input[:, i] * (expert + 1), output[:, i])


@pytest.mark.mpi
def test_forward_routing_shuffle(device=rank):
    dgrid = DistributionGrid(expert_parallel_group_size = dist.get_world_size())
    model_dim = 6
    num_local_experts = 4
    num_rank = dist.get_world_size(dist.group.WORLD)
    num_experts = num_rank * num_local_experts
    input = torch.ones(4 * num_local_experts, 32, model_dim).to(device)
    gate = RoundRobinGate(model_dim, num_experts, dgrid)
    experts = torch.nn.ModuleList()
    for i in range(num_local_experts):
        expert = torch.nn.Linear(model_dim, model_dim, bias=False)
        # Use scaling matrix (each rank has a different scale)
        scale = dist.get_rank() * num_local_experts + i + 1
        expert.weight = torch.nn.Parameter(torch.eye(model_dim) * scale)
        experts.append(expert)
    options = {"enable_base_layer_shuffling" : True}
    moe = MixtureOfExpertsFunc(gate, experts, distribution_grid=dgrid, options=options).to(device)
    if use_ort:
        moe = ORTModule(moe)
    rand_fixed_idx = num_rank // 2 - 1
    rank_lists = [list(range(0, rand_fixed_idx)), list(range(rand_fixed_idx, num_rank))]
    for r in rank_lists:
        tmp = dist.new_group(r)
        if rank in r:
            pg = tmp
    output = moe(input, shuffle_group=pg)
    if use_ort:
        loss = output[0].sum()
        loss.backward()
    assert output.shape == input.shape


@pytest.mark.mpi
@pytest.mark.with_ort
def test_backward(device=rank):
    dgrid = DistributionGrid(expert_parallel_group_size = dist.get_world_size())
    loss = torch.nn.MSELoss()
    model_dim = 8
    num_experts = dist.get_world_size(dist.group.WORLD)
    input = torch.randn(4, 16, model_dim).to(device)
    gate = TopKGate(model_dim, num_experts, k=2, dgrid=dgrid)
    experts = torch.nn.ModuleList()
    expert = torch.nn.Linear(model_dim, model_dim, bias=False)
    experts.append(expert)
    # Use identity matrix
    expert.weight = torch.nn.Parameter(torch.eye(model_dim))
    moe = MixtureOfExpertsFunc(gate, experts, distribution_grid=dgrid).to(device)
    if use_ort:
        moe = ORTModule(moe)
    output = moe(input)
    assert output.shape == input.shape
    output = loss(output, input)
    output.backward()
    assert torch.allclose(expert.weight.grad, torch.zeros_like(expert.weight))

#Construct two models, one with padding-0 explictly, another with padding implicitly by all-to-all. The gradents computed by bwd should be the same
@pytest.mark.with_ort1
def test_all2all(device=rank):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ffn0 = torch.nn.Linear(4, 10, bias=False)
            self.ffn1 = torch.nn.Linear(10, 4, bias=False)
            i = 0
            for p in self.ffn0.parameters():
                torch.manual_seed(i)
                torch.nn.init.normal_(p)
                i+= 1
            i = 0
            for p in self.ffn1.parameters():
                torch.manual_seed(i)
                torch.nn.init.normal_(p)
                i+= 1
        def forward(self, x):
            x = self.ffn0(x)
            x = AllToAll.apply(dist.group.WORLD, x, 8)
            x = self.ffn1(x)
            return x
    loss = torch.nn.MSELoss()
    model_0 = Model().to(rank)
    model_1 = Model().to(rank)
    if use_ort:
        model_0 = ORTModule(model_0)
        model_1 = ORTModule(model_1)

    if rank == 0:
        x0 = torch.arange(8*6*4, dtype=torch.float32).reshape(8, 6, 4).to(rank)
        x1 = torch.nn.functional.pad(x0, pad=(0, 0, 0, 2), mode='constant', value=0.0).to(rank)
    else:
        x0 = torch.arange(8*6*4, dtype=torch.float32).reshape(8, 6, 4).to(rank)
        x0 = torch.nn.functional.pad(x0, pad=(0, 0, 0, 2), mode='constant', value=0.0).to(rank)
        x1 = x0.clone().detach().to(rank)

    output0 = model_0(x0)

    label = torch.zeros_like(output0)
    output0 = loss(output0, label)
    output0.backward()

    output1 = model_1(x1)
    output1 = loss(output1, label)
    output1.backward()

    for p0, p1 in zip(model_0.parameters(), model_1.parameters()):
        assert torch.allclose(p0.grad, p1.grad)

def test_moe_reset_state1():
    dgrid = DistributionGrid(expert_parallel_group_size = dist.get_world_size())
    model_dim = 8
    num_experts = 4
    gate = TopKGate(model_dim, num_experts, k=2, dgrid=dgrid)
    experts = torch.nn.ModuleList()
    expert = torch.nn.Linear(model_dim, model_dim, bias=False)
    experts.append(expert)
    moe0 = MixtureOfExpertsFunc(gate, experts, distribution_grid = dgrid)
    moe0.enc_max_len["need_update"] = False
    #When calling del moe0, the reset_moe_state is called, and "need_update" is reset to true
    del moe0
    assert MixtureOfExperts.enc_max_len["need_update"] is True

@pytest.mark.mpi
@pytest.mark.with_ort
def test_moe_reset_state2(device=rank):
    dgrid = DistributionGrid(expert_parallel_group_size = dist.get_world_size())
    model_dim = 8
    num_experts = dist.get_world_size(dist.group.WORLD)
    gate = TopKGate(model_dim, num_experts, k=2, dgrid=dgrid)
    experts = torch.nn.ModuleList()
    expert = torch.nn.Linear(model_dim, model_dim, bias=False)
    experts.append(expert)
    moe1 = MixtureOfExpertsFunc(gate, experts, distribution_grid = dgrid).to(device)
    if use_ort:
        moe1 = ORTModule(moe1)

    def hook_fun(self, input, output):
        MixtureOfExperts.reset_moe_state()

    #registering the reset_moe_state as fwd hook, after fwd call, it should be called to reset the state
    moe1.register_forward_hook(hook_fun)

    input = torch.rand(4, 16, model_dim).to(device)
    output = moe1(input)
    if use_ort:
        loss = output[0].sum()
        loss.backward()
    assert moe1.enc_max_len["need_update"] is True

#Test whether the merged FFN and FFN experts have the same output
@pytest.mark.with_ort1
def test_merged_ffn_expert():
    dgrid = DistributionGrid()
    experts = torch.nn.ModuleList()
    num_experts = 4
    d_model = 2
    d_ff = 16
    group_size = 32
    capacity = 64

    torch.manual_seed(42)
    for n in range(num_experts):
        exp = FFNExpert(d_model, d_ff, dgrid = dgrid)
        experts+= [exp]
    inputs= torch.randn(group_size, num_experts, capacity, d_model)
    output_list = []
    input_list = inputs.chunk(num_experts, dim=1)

    for input, exp in zip(input_list, experts):
        output_list += [exp(input)]
    output_ref = torch.cat(output_list, dim=1)

    torch.manual_seed(42)
    merged_experts = MergedFFNExpert(d_model, d_ff, num_experts, dgrid=dgrid)
    if use_ort:
        merged_experts = ORTModule(merged_experts)
    output_merged = merged_experts(inputs)
    if use_ort:
        loss = output_merged[0].sum()
        loss.backward()

    # The default setting for torch.allclose() is atol=1e-08, whilch will trigger assert failed for A100 GPU. Reset atol=1e-07 to fix the failed for A100 GPU.
    assert torch.allclose(output_ref, output_merged, atol=1e-07)

@pytest.mark.with_ort
def test_expert_moez_encoder():
    dgrid = DistributionGrid(expert_parallel_group_size = dist.get_world_size())
    d_model = 512
    nhead = 8
    nlang = 4
    dim_feedforward = 256
    num_local_experts = 4
    nexperts = dist.get_world_size(dist.group.WORLD) * num_local_experts
    gate = TopKGate(d_model, nexperts, k=1, dgrid=dgrid)  # top-1
    shared_encoder = TransformerMoEEncoderLayer(d_model, nhead, dim_feedforward=dim_feedforward,
                                                nexperts=nexperts, gate=gate, distribution_grid=dgrid)
    for p in shared_encoder.named_parameters():
        print("shared = ", p[0], p[1].numel())
    total_shared_paramters = sum(p.numel() for p in shared_encoder.parameters())

    expected_shared_parameters = d_model * nexperts  # number of parameters in gate
    expected_shared_parameters += num_local_experts * 2 * d_model * \
        dim_feedforward  # number of parameters in Experts
    # number of attention and the W_o
    expected_shared_parameters += 4 * (d_model * d_model + d_model)
    expected_shared_parameters += 4 * d_model  # number of layernorms

    dgrid2 = DistributionGrid(expert_parallel_group_size = dist.get_world_size())

    language_encoder = LanguageExpertMoEEncoderLayer(d_model, nhead, dim_feedforward=dim_feedforward,
                                                    nexperts=nexperts, gate=gate,
                                                    nlang_experts = nlang, distribution_grid = dgrid2)
    if use_ort:
        language_encoder = ORTModule(language_encoder)
    for p in language_encoder.named_parameters():
        print("lang = ", p[0], p[1].numel())
    total_lang_paramters = sum(p.numel() for p in language_encoder.parameters())

    expected_lang_parameters = d_model * nexperts  # number of parameters in gate
    expected_lang_parameters += num_local_experts * 2 * d_model * \
        dim_feedforward * nlang # number of parameters in Experts
    # number of attention and the W_o
    expected_lang_parameters += 4 * (d_model * d_model + d_model) * nlang
    expected_lang_parameters += 4 * d_model * nlang # number of layernorms

    assert total_shared_paramters == expected_shared_parameters
    assert total_lang_paramters == expected_lang_parameters

@pytest.mark.with_ort
def test_expert_moez_decoder():
    dgrid = DistributionGrid(expert_parallel_group_size = dist.get_world_size())
    d_model = 512
    nhead = 8
    nlang = 4
    dim_feedforward = 256
    num_local_experts = 4
    nexperts = dist.get_world_size(dist.group.WORLD) * num_local_experts
    gate = TopKGate(d_model, nexperts, k=1, dgrid=dgrid)  # top-1
    shared_decoder = TransformerMoEDecoderLayer(d_model, nhead, dim_feedforward=dim_feedforward,
                                                nexperts=nexperts, gate=gate, distribution_grid=dgrid)
    if use_ort:
        shared_decoder = ORTModule(shared_decoder)
    total_shared_paramters = sum(p.numel() for p in shared_decoder.parameters())

    expected_shared_parameters = d_model * nexperts  # number of parameters in gate
    expected_shared_parameters += num_local_experts * 2 * d_model * \
        dim_feedforward  # number of parameters in Experts
    # number of attention and the W_o
    expected_shared_parameters += 2 * 4 * (d_model * d_model + d_model)
    expected_shared_parameters += 6 * d_model  # number of layernorms

    dgrid2 = DistributionGrid(expert_parallel_group_size = dist.get_world_size())
    language_decoder = LanguageExpertMoEDecoderLayer(d_model, nhead, dim_feedforward=dim_feedforward,
                                                    nexperts=nexperts, gate=gate,
                                                    nlang_experts=nlang, distribution_grid=dgrid2)
    if use_ort:
        language_decoder = ORTModule(language_decoder)
    total_lang_paramters = sum(p.numel() for p in language_decoder.parameters())

    expected_lang_parameters = d_model * nexperts  # number of parameters in gate
    expected_lang_parameters += num_local_experts * 2 * d_model * \
        dim_feedforward * nlang # number of parameters in Experts
    # number of attention and the W_o
    expected_lang_parameters += 2 * 4 * (d_model * d_model + d_model) * nlang
    expected_lang_parameters += 6 * d_model * nlang # number of layernorms

    assert total_shared_paramters == expected_shared_parameters
    assert total_lang_paramters == expected_lang_parameters

@pytest.mark.with_ort
def test_moe_allreduce(device=rank):
    dgrid = DistributionGrid(expert_parallel_group_size = dist.get_world_size())
    num_local_experts = 4
    num_experts = dist.get_world_size(dist.group.WORLD) * num_local_experts
    d_model = 2
    d_ff = 16
    gating_fn = TopKGate(d_model, num_experts, k=1, dgrid=dgrid)  # Top1
    merged_experts = MergedFFNExpert(d_model, d_ff, num_local_experts, dgrid=dgrid)
    model = MixtureOfExpertsFunc(gating_fn, merged_experts, dgrid).to(device)
    if use_ort:
        model = ORTModule(model)
    for param in model.parameters():
        param.grad = torch.ones(param.size(), dtype=param.dtype, device=param.device) * rank
    moe_module_all_reduce(model, dgrid)

    #check the result
    no_moe_param_size = 0
    no_moe_param_size_ref= d_model * num_experts
    for param in model.parameters():
        param_grad_ref = torch.ones(param.size(), dtype=param.dtype, device=param.device)
        if not is_moe_parameter(param):
            param_grad_ref *= sum(range(0, size))/size
            assert torch.allclose(param.grad, param_grad_ref)
            no_moe_param_size += param.numel()
        else:
            param_grad_ref *= rank
            assert torch.equal(param.grad, param_grad_ref)
    assert no_moe_param_size==no_moe_param_size_ref


# Constucts 2 way dp on 2 way ep parallelism
# tests whether all_reduce on gradients are yielding right values
@pytest.mark.with_ort
def test_dp_group_allreduce(device=rank):
    assert dist.get_world_size() >= 2, "Need at least 2 gpus to test this"
    num_experts = dist.get_world_size(dist.group.WORLD)
    dp_ways = 2

    # configure groups
    ep_ways = dist.get_world_size() // dp_ways
    rank_list = [ i for i in range(0, dist.get_world_size())]
    dgrid = DistributionGrid(expert_parallel_group_size = ep_ways, expert_parallel_replica_group_size = dp_ways)

    d_model = 2
    d_ff = 16
    gating_fn = TopKGate(d_model, num_experts, k=1, dgrid=dgrid)  # Top1
    expert = FFNExpert(d_model, d_ff, dgrid=dgrid)
    experts = torch.nn.ModuleList()
    experts.append(expert)
    model = MixtureOfExpertsFunc(gating_fn, experts, distribution_grid=dgrid).to(device)
    if use_ort:
        model = ORTModule(model)

    # set rank as the the value of gradients
    for param in model.parameters():
        param.grad = torch.ones(param.size(), dtype=param.dtype, device=param.device) * rank

    moe_module_all_reduce(model, dgrid)

    # get the gradients
    dp_grads = []
    global_grads = []
    for param in model.parameters():
        if not is_moe_parameter(param):
            global_grads.append(param.grad.view(-1))
        else:
            dp_grads.append(param.grad.view(-1))

    # now we should be able to see right value in reduced grads
    # non-moe parameters
    import numpy as np
    global_exp = sum(rank_list)/len(rank_list)
    for grad in global_grads:
        dummy = torch.full_like(grad, global_exp)
        assert(torch.allclose(grad, dummy))

    # moe parameters
    start_rank = dgrid.get_expert_replica_src_rank()
    expected = sum(rank_list[start_rank::ep_ways])/dp_ways # if rank%ep_ways == 0 # else sum(rank_list[1::ep_ways])/dp_ways
    for grad in dp_grads:
        dummy = torch.full_like(grad, expected)
        assert torch.allclose(grad, dummy)


# vary the dp_ways while keeping the number of experts constant
# perform the sanity check on gradient all_reduce for expert and non-expert parameters
@pytest.mark.mpi
@pytest.mark.with_ort
def test_dp_group_with_all2all(device = rank):
    dp_ways = 1
    num_experts = dist.get_world_size(dist.group.WORLD)
    while dp_ways <= dist.get_world_size():
        # initialize the grid
        ep_ways = dist.get_world_size() // dp_ways

        dgrid = DistributionGrid(expert_parallel_group_size = ep_ways, expert_parallel_replica_group_size = dp_ways)

        # build a simple moe layer
        loss = torch.nn.MSELoss()
        num_local_experts = num_experts // ep_ways
        assert num_experts%ep_ways == 0
        d_model = 8

        input = torch.randn(4, 16, d_model).to(device)
        gating_fn = TopKGate(d_model, num_experts, k=2, dgrid=dgrid)  # Top1
        experts = torch.nn.ModuleList()
        for _ in range(num_local_experts):
            expert = torch.nn.Linear(d_model, d_model, bias = False)
            experts.append(expert)

        expert.weight = torch.nn.Parameter(torch.randn(d_model, d_model))
        moe = MixtureOfExpertsFunc(gating_fn, experts, distribution_grid= dgrid).to(device)
        if use_ort:
            moe = ORTModule(moe)
        output = moe(input.float())

        # make sure the shape of input and output are same
        assert output.shape == input.shape

        # backpropagation
        output = loss(output, input)
        output.backward()

        moe_module_all_reduce(moe, dgrid)

        #gradient checking: pull out the grads
        dp_grads = []
        global_grads = []
        dp_grads_pre = []
        global_grads_pre = []
        for param in moe.parameters():
            if not is_moe_parameter(param):
                global_grads_pre.append(param.grad.clone().detach())
                global_grads.append(param.grad.data)
            else:
                dp_grads.append(param.grad.data)
                dp_grads_pre.append(param.grad.clone().detach())

        #1. expert gradients among dp_group should be same
        if dp_ways >= 2:
            for grad, grad_exp in zip(dp_grads, dp_grads_pre):
                dist.all_reduce(grad, op = dist.ReduceOp.SUM, group = dgrid.get_expert_replica_group())
                grad = torch.div(grad, dgrid.get_expert_replica_world_size())
                assert torch.allclose(grad_exp, grad)

        #2. non expert gradients among world should be same
        for grad, grad_exp in zip(global_grads, global_grads_pre):
            dist.all_reduce(grad, op = dist.ReduceOp.SUM, group = torch.distributed.group.WORLD)
            grad = torch.div(grad, torch.distributed.get_world_size())
            assert torch.allclose(grad_exp, grad)

        dp_ways *= 2


# when all expert weights are unit matrix, input should be reproduced at the end of moe layer if all2all is working
# TODO: all2all check on gradient
@pytest.mark.mpi
@pytest.mark.with_ort
def test_ep_group_all2all_forward(device = rank):
    dp_ways = 1
    num_experts = dist.get_world_size(dist.group.WORLD)
    while dp_ways <= dist.get_world_size():
        #initialize the grid
        ep_ways = dist.get_world_size() // dp_ways

        dgrid = DistributionGrid(expert_parallel_group_size = ep_ways, expert_parallel_replica_group_size = dp_ways)

        # build a simple moe layer
        loss = torch.nn.MSELoss()
        num_local_experts = num_experts // ep_ways
        assert num_experts%ep_ways == 0
        d_model = 8

        input = torch.randn(4, 16, d_model).to(device)
        gating_fn = TopKGate(d_model, num_experts, k=2, dgrid=dgrid)  # Top1
        experts = torch.nn.ModuleList()
        for _ in range(num_local_experts):
            expert = torch.nn.Linear(d_model, d_model, bias = False)
            expert.weight = torch.nn.Parameter(torch.eye(d_model))
            experts.append(expert)

        moe = MixtureOfExpertsFunc(gating_fn, experts, distribution_grid = dgrid).to(device)
        if use_ort:
            moe = ORTModule(moe)
        output = moe(input)
        if use_ort:
            loss = output[0].sum()
            loss.backward()

        # make sure the input and output are equal
        assert torch.allclose(output, input)

        dp_ways *= 2

@pytest.mark.with_ort1
def test_ep_group_all2all_backward(device=rank):
    if dist.get_world_size() != 4:
        return 0
    # ran with dp_ways = 1 and world_size = 2, twice
    # On first set weight to eye()*2 and eye()*3, on second run set weight to eye()*4 and eye()*5, save the gradient
    # load the saved grads
    # NOTE: Since the result is hard coded, it only works for 4 ranks.
    world_grads = [torch.tensor([[12.5000, 12.5000, 12.5000, 12.5000, 12.5000, 12.5000, 12.5000, 12.5000],
        [12.5000, 12.5000, 12.5000, 12.5000, 12.5000, 12.5000, 12.5000, 12.5000],
        [12.5000, 12.5000, 12.5000, 12.5000, 12.5000, 12.5000, 12.5000, 12.5000],
        [12.5000, 12.5000, 12.5000, 12.5000, 12.5000, 12.5000, 12.5000, 12.5000],
        [12.5000, 12.5000, 12.5000, 12.5000, 12.5000, 12.5000, 12.5000, 12.5000],
        [12.5000, 12.5000, 12.5000, 12.5000, 12.5000, 12.5000, 12.5000, 12.5000],
        [12.5000, 12.5000, 12.5000, 12.5000, 12.5000, 12.5000, 12.5000, 12.5000],
        [12.5000, 12.5000, 12.5000, 12.5000, 12.5000, 12.5000, 12.5000, 12.5000]]),
        torch.tensor([[1.0394e-05, 1.0394e-05, 1.0394e-05, 1.0394e-05, 1.0394e-05, 1.0394e-05, 1.0394e-05, 1.0394e-05],
        [1.0394e-05, 1.0394e-05, 1.0394e-05, 1.0394e-05, 1.0394e-05, 1.0394e-05, 1.0394e-05, 1.0394e-05],
        [1.0394e-05, 1.0394e-05, 1.0394e-05, 1.0394e-05, 1.0394e-05, 1.0394e-05, 1.0394e-05, 1.0394e-05],
        [1.0394e-05, 1.0394e-05, 1.0394e-05, 1.0394e-05, 1.0394e-05, 1.0394e-05, 1.0394e-05, 1.0394e-05],
        [1.0394e-05, 1.0394e-05, 1.0394e-05, 1.0394e-05, 1.0394e-05, 1.0394e-05, 1.0394e-05, 1.0394e-05],
        [1.0394e-05, 1.0394e-05, 1.0394e-05, 1.0394e-05, 1.0394e-05, 1.0394e-05, 1.0394e-05, 1.0394e-05],
        [1.0394e-05, 1.0394e-05, 1.0394e-05, 1.0394e-05, 1.0394e-05, 1.0394e-05, 1.0394e-05, 1.0394e-05],
        [1.0394e-05, 1.0394e-05, 1.0394e-05, 1.0394e-05, 1.0394e-05, 1.0394e-05, 1.0394e-05, 1.0394e-05]]),
        torch.tensor([[37.5000, 37.5000, 37.5000, 37.5000, 37.5000, 37.5000, 37.5000, 37.5000],
        [37.5000, 37.5000, 37.5000, 37.5000, 37.5000, 37.5000, 37.5000, 37.5000],
        [37.5000, 37.5000, 37.5000, 37.5000, 37.5000, 37.5000, 37.5000, 37.5000],
        [37.5000, 37.5000, 37.5000, 37.5000, 37.5000, 37.5000, 37.5000, 37.5000],
        [37.5000, 37.5000, 37.5000, 37.5000, 37.5000, 37.5000, 37.5000, 37.5000],
        [37.5000, 37.5000, 37.5000, 37.5000, 37.5000, 37.5000, 37.5000, 37.5000],
        [37.5000, 37.5000, 37.5000, 37.5000, 37.5000, 37.5000, 37.5000, 37.5000],
        [37.5000, 37.5000, 37.5000, 37.5000, 37.5000, 37.5000, 37.5000, 37.5000]]),
        torch.tensor([[3.1182e-05, 3.1182e-05, 3.1182e-05, 3.1182e-05, 3.1182e-05, 3.1182e-05, 3.1182e-05, 3.1182e-05],
        [3.1182e-05, 3.1182e-05, 3.1182e-05, 3.1182e-05, 3.1182e-05, 3.1182e-05, 3.1182e-05, 3.1182e-05],
        [3.1182e-05, 3.1182e-05, 3.1182e-05, 3.1182e-05, 3.1182e-05, 3.1182e-05, 3.1182e-05, 3.1182e-05],
        [3.1182e-05, 3.1182e-05, 3.1182e-05, 3.1182e-05, 3.1182e-05, 3.1182e-05, 3.1182e-05, 3.1182e-05],
        [3.1182e-05, 3.1182e-05, 3.1182e-05, 3.1182e-05, 3.1182e-05, 3.1182e-05, 3.1182e-05, 3.1182e-05],
        [3.1182e-05, 3.1182e-05, 3.1182e-05, 3.1182e-05, 3.1182e-05, 3.1182e-05, 3.1182e-05, 3.1182e-05],
        [3.1182e-05, 3.1182e-05, 3.1182e-05, 3.1182e-05, 3.1182e-05, 3.1182e-05, 3.1182e-05, 3.1182e-05],
        [3.1182e-05, 3.1182e-05, 3.1182e-05, 3.1182e-05, 3.1182e-05, 3.1182e-05, 3.1182e-05, 3.1182e-05]])]

    # run experiment with world_size = 4, dp_ways = 2, weight to eye()*(rank+2)
    # expert weight grads need to match with saved values
    dp_ways = 2
    ep_ways = dist.get_world_size() // dp_ways
    dgrid = DistributionGrid(expert_parallel_group_size = ep_ways, expert_parallel_replica_group_size = dp_ways)

    num_experts = 2# dist.get_world_size(dist.group.WORLD)
    loss = torch.nn.MSELoss()
    model_dim = 8
    input = (torch.ones(4, 16, model_dim)*5).to(device)

    gate = TopKGate(model_dim, num_experts, k=2, dgrid=dgrid)
    # set gate weights always same, so we obtain deterministic mapping
    gate.wg.weight = torch.nn.Parameter(torch.tensor([[ 0.9311,  0.1706,  0.3681,  1.7191,  2.0357,  0.2269, -0.0920, -0.2983],
        [ 1.2228,  0.3268, -0.0645, -0.7305, -1.1829, -0.0968,  1.2634,  1.5229]]))

    experts = torch.nn.ModuleList()
    expert = torch.nn.Linear(model_dim, model_dim, bias=False)
    experts.append(expert)

    # Use identity matrix multiplied with rank+2, avoid identity and zero matrix
    expert.weight = torch.nn.Parameter(torch.eye(model_dim)*(device+2))
    moe = MixtureOfExpertsFunc(gate, experts, distribution_grid=dgrid).to(device)
    if use_ort:
        moe = ORTModule(moe)

    output = moe(input)

    output = loss(output, input)
    output.backward()
    for param in moe.parameters():
        if not is_moe_parameter(param):
            # grad should be same in every rank
            local_grad = param.grad.data.detach()
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.MAX, group = dist.group.WORLD)
            assert torch.allclose(local_grad, param.grad.data)
        else:
            # grad should match with saved value
            exp_grad = world_grads[device].to(device)
            assert torch.allclose(param.grad.data, exp_grad)


# make sure the max-len computation doing allreduce among ep_group is working correctly
# set arbitrary values to max_len at the beginning, make sure they sync after function call
@pytest.mark.with_ort
def test_max_len_all_reduce(device = rank):
    dp_ways = 1
    num_experts = dist.get_world_size(dist.group.WORLD)
    while dp_ways <= dist.get_world_size():
        # Initialize the grid
        ep_ways = dist.get_world_size() // dp_ways
        rank_list = list(range(0, dist.get_world_size()))
        dgrid = DistributionGrid(expert_parallel_group_size = ep_ways, expert_parallel_replica_group_size = dp_ways)

        ep_rank_list = [rank_list[i:i+ep_ways] for i in range(0, len(rank_list), ep_ways)]


        # put 2 expert per gpu, that way 'e' dimension on dispatched input varies as we change dp_ways
        num_local_experts = 2
        assert num_experts%ep_ways == 0
        d_model = 8

        input = torch.randn(4, 4, d_model).to(device)
        gating_fn = TopKGate(d_model, ep_ways*num_local_experts, k=1, capacity_factor=1.25, dgrid=dgrid)  # Top1
        experts = torch.nn.ModuleList()
        for _ in range(num_local_experts):
            expert = torch.nn.Linear(d_model, d_model, bias = False)
            expert.weight = torch.nn.Parameter(torch.eye(d_model))
            experts.append(expert)
        moe = MixtureOfExpertsFunc(gating_fn, experts, distribution_grid= dgrid).to(device)
        if use_ort:
            moe = ORTModule(moe)

        # set max len arbitrarily high so that we know allreduce is working
        max_len = {"max_len": rank * 16, "capacity_fp": 0, "need_update": True}
        print("Pre: Rank = {}, max_len = {}".format(device, max_len["max_len"]))

        # Reshape input by dropping sequence dimension.
        reshaped_input = input.reshape(-1, d_model)
        _, dispatch_mask, capacity_fp = moe.gate(reshaped_input)
        dispatched_input = reshaped_input[dispatch_mask % reshaped_input.shape[0]] #dim [E, C, M], "%" is needed to round the (token_idx + num_tokens*topk) to original tokens for indicing
        dispatched_input = dispatched_input.reshape(-1, math.ceil(capacity_fp), d_model)
        c_cpu = dispatched_input.shape[1]
        moe.get_max_len(c_cpu, max_len, dispatched_input.get_device())

        print("Post: Rank = {}, max_len = {}, c_cpu = {}".format(device, max_len["max_len"], c_cpu))
        assert max_len["max_len"] >= c_cpu

        #get max_len for each worker in ep_group and make sure its same in every machine
        max_len_copy = max_len["max_len"]

        from mpi4py import MPI
        mpi_ep_group = None
        for ep_list in ep_rank_list:
            tmp = MPI.COMM_WORLD.Create_group(MPI.COMM_WORLD.group.Incl(ep_list))
            if device in ep_list:
                mpi_ep_group = tmp
        max_len_copy = mpi_ep_group.allreduce(max_len_copy, MPI.MAX)
        assert max_len_copy == max_len["max_len"]

        dp_ways *= 2

@pytest.mark.mpi
def test_mp_gating(device = rank):
    torch.manual_seed(7)
    input = torch.rand(4, 10).to(device) #SM
    wg = torch.nn.Linear(10, 2).to(device)

    dispatch_mask_ref = torch.tensor([1, 2, 3, 0]).to(device)
    expert_cumsum_ref = torch.tensor([0, 3, 4])

    dgrid = DistributionGrid(expert_slicing_group_size = 2, expert_parallel_replica_group_size= dist.get_world_size()//2)

    gate = TopKGate(10, 4, dgrid, k=1).to(device)
    if use_ort:
        gate = ORTModule(gate)
    gate.wg = wg
    outputs = gate(input)
    if use_ort:
        loss = outputs[0].sum()
        loss.backward()

    assert torch.equal(outputs[1], dispatch_mask_ref)
    assert(torch.equal(outputs[2], expert_cumsum_ref))

def do_test_mp_moe_forward(dynamic_capacity, device = rank):
    dgrid = DistributionGrid(expert_slicing_group_size = 2, expert_parallel_replica_group_size=dist.get_world_size()//2)

    torch.manual_seed(7)

    d_model = 10
    d_ff = d_model
    num_expert = 4
    d_token = 6
    options = {}
    options["enable_dynamic_capacity"] = dynamic_capacity
    gate = TopKGate(d_model, num_expert, dgrid, k=1, options=options).to(device)

    experts = torch.nn.ModuleList()

    for i in range(num_expert):
        expert = FFNExpert(d_model, d_ff, dgrid).to(device)
        expert.linear1.weight = torch.nn.Parameter(torch.eye(d_model) * (device%2+1))
        expert.linear2.weight = torch.nn.Parameter(torch.eye(d_model) * (device%2+1))
        experts.append(expert)

    moe = MixtureOfExpertsFunc(gate, experts, distribution_grid=dgrid, options=options).to(device)
    if use_ort:
        moe = ORTModule(moe)

    input = torch.rand(1, d_token, d_model).to(device)
    output = moe(input)
    if use_ort:
        loss = output[0].sum()
        loss.backward()

    assert dgrid.get_expert_slicing_world_size() == 2
    if dynamic_capacity:
        return

    #compute the reference gate_s by rerun the gate function with the same input
    reshaped_input_ref = torch.cat((input, input), 0).reshape(-1, d_model)
    combine_weights_ref, dispatch_mask, expert_cumsum = gate(reshaped_input_ref)

    output_ref = torch.einsum("ks, ksm->sm", combine_weights_ref[:, device%2*d_token: (device%2+1)*d_token], input * 5.0)
    assert torch.allclose(output_ref, output, atol = 1e-03)

def test_mp_moe_forward_pass():
    do_test_mp_moe_forward(dynamic_capacity=True)
    do_test_mp_moe_forward(dynamic_capacity=False)

#Compare the MP forward and backward pass by: (1) ref model: running the entire model on one GPU,
# (2) testing_model: running the same model on multiple GPUs, the forward and backward output should be the same (allclose)
def test_mp_moe_forward_backward(device = rank):
    torch.manual_seed(7)
    dgrid = DistributionGrid(expert_slicing_group_size = dist.get_world_size())
    dgrid_ref = DistributionGrid(expert_slicing_group_size = 1)

    d_model = 10
    d_ff = 20
    num_expert = 4
    global_d_token = 12
    world_size = dist.get_world_size()
    local_d_token = global_d_token // world_size

    gate_ref = TopKGate(d_model, num_expert, dgrid_ref, k=1).to(device)

    gate = TopKGate(d_model, num_expert, dgrid, k=1).to(device)
    weight_ref = gate_ref.wg.weight
    gate.wg.weight = torch.nn.Parameter(weight_ref.detach().clone())

    experts_ref = torch.nn.ModuleList()
    for i in range(num_expert):
        expert_ref = FFNExpert(d_model, d_ff, dgrid_ref).to(device)
        experts_ref.append(expert_ref)

    experts = torch.nn.ModuleList()
    for i in range(num_expert):
        expert = FFNExpert(d_model, d_ff, dgrid).to(device)
        w1_ref = experts_ref[i].linear1.weight
        expert.linear1.weight = torch.nn.Parameter(w1_ref.detach().clone()[rank*d_ff//world_size:(rank+1)*d_ff//world_size])
        w2_ref = experts_ref[i].linear2.weight
        expert.linear2.weight = torch.nn.Parameter(w2_ref.detach().clone()[:, rank*d_ff//world_size:(rank+1)*d_ff//world_size])
        experts.append(expert)

    moe_ref = MixtureOfExpertsFunc(gate_ref, experts_ref, distribution_grid=dgrid_ref)
    moe = MixtureOfExpertsFunc(gate, experts, distribution_grid = dgrid)

    input_ref = torch.rand(1, global_d_token, d_model).to(device)
    non_padding_ref = (torch.rand(1, global_d_token) > 0.1).float()

    input = input_ref.detach().clone()[:, rank*local_d_token:(rank+1) * local_d_token, :].to(device)
    non_padding = non_padding_ref.detach().clone()[:, rank*local_d_token:(rank+1)*local_d_token].to(device)

    output_ref = moe_ref(input_ref, non_padding = non_padding_ref)
    moe_ref.reset_moe_encoder_state()
    output = moe(input, non_padding = non_padding)
    moe.reset_moe_encoder_state()

    #The output of the forward pass should be identical compare to it is run on single machine
    assert torch.allclose(output_ref[:, rank*local_d_token:(rank+1)*local_d_token, :], output)

    loss_ref = torch.nn.MSELoss(reduction='sum')
    loss = torch.nn.MSELoss(reduction='sum')

    output_ref = loss_ref(output_ref, input_ref)
    output = loss(output, input)

    output_ref.backward()
    output.backward()

    for i in range(num_expert):
        assert torch.allclose(experts[i].linear1.weight.grad,
        experts_ref[i].linear1.weight.grad[rank*d_ff//world_size:(rank+1)*d_ff//world_size])

        assert torch.allclose(experts[i].linear2.weight.grad,
        experts_ref[i].linear2.weight.grad[:, rank*d_ff//world_size:(rank+1)*d_ff//world_size])

    #After allreduce, the gate_ref.wg.weight.grad should be the same as gate.wg.weight.grad
    dist.all_reduce(gate.wg.weight.grad.data, op=dist.ReduceOp.SUM)
    assert torch.allclose(gate_ref.wg.weight.grad, gate.wg.weight.grad)

#NOTE: We need to put all apex tests at the bottom, otherwise all tests after the apex tests are all casted to fp16.
#Test the manually casting alltoall input to fp16 is the same as not casting, when apex O1 is applied
#@pytest.mark.with_ort # opened an exporter issue for investigation
def test_moe_fp16(device = rank):
    model_dim = 256
    ff_dim = 64

    num_local_experts = 4
    num_experts = dist.get_world_size(dist.group.WORLD) * num_local_experts

    dgrid_ref = DistributionGrid(expert_parallel_group_size = dist.get_world_size())
    options = { "fp16_mode" : True}
    gating_fn = TopKGate(model_dim, num_experts, k=1, switch_jitter=0.0, dgrid=dgrid_ref, options = options)  # Top1
    merged_experts = MergedFFNExpert(model_dim, ff_dim, num_local_experts, dgrid=dgrid_ref)

    model_ref = MixtureOfExpertsFunc(gating_fn, merged_experts, distribution_grid=dgrid_ref).to(device)
    if use_ort:
        model_ref = ORTModule(model_ref)
    optimizer_ref = torch.optim.SGD(model_ref.parameters(), lr=1e-3)

    dgrid = DistributionGrid(expert_parallel_group_size = dist.get_world_size())
    options = { "fp16_mode" : True}
    model = MixtureOfExpertsFunc(gating_fn, merged_experts, distribution_grid = dgrid, options = options).to(device)
    if use_ort:
        model = ORTModule(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    input = torch.rand(4, 16, model_dim).to(device)
    model_ref, optimizer_ref = apex_amp.initialize(model_ref, optimizer_ref, opt_level="O1")
    output_ref = model_ref(input)

    model, optimizer = apex_amp.initialize(model, optimizer, opt_level="O1")
    output = model(input)
    if use_ort:
        loss = output[0].sum()
        loss.backward()

    assert torch.equal(output_ref, output)

#@pytest.mark.with_ort # opened an exporter issue for investigation
def test_moe_loss_scale(device = rank):
    model_dim = 256
    ff_dim = 64
    num_local_experts = 4
    dgrid = DistributionGrid(expert_parallel_group_size = dist.get_world_size())
    merged_experts = MergedFFNExpert(model_dim, ff_dim, num_local_experts, dgrid=dgrid)
    num_experts = dist.get_world_size(dist.group.WORLD) * num_local_experts
    options = { "fp16_mode" : True}
    gating_fn = TopKGate(model_dim, num_experts, k=1, dgrid=dgrid, options = options)  # Top1
    model = MixtureOfExpertsFunc(gating_fn, merged_experts, distribution_grid = dgrid, options = options).to(device)
    if use_ort:
        model = ORTModule(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    input = torch.rand(4, 16, model_dim).to(device)
    model, optimizer = apex_amp.initialize(model, optimizer, opt_level="O1")
    output = model(input)
    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(output, input)

    apex_amp_scale_check_overflow_override(apex_amp)
    with apex_amp.scale_loss(loss, optimizer) as scaled_loss:
        loss_scale_old = apex_amp._amp_state.loss_scalers[0].loss_scale()
        scaled_loss.backward()
        dgrid = DistributionGrid(expert_parallel_group_size = dist.get_world_size())
        moe_module_all_reduce(model, dgrid)
        #force some gradient in the rank 0 to be NaN
        if dist.get_rank() == 0:
            for param in model.parameters():
                param.grad.data[0] = float('nan')
                break
    optimizer.step()
    loss_scale_new = apex_amp._amp_state.loss_scalers[0].loss_scale()
    #check whether the overflow flag is broadcasted to all ranks
    assert loss_scale_new * 2 == loss_scale_old, f"{torch.distributed.get_rank()} failed"

# sanity check to make sure weights are synchronized at initialization
@pytest.mark.mpi
@pytest.mark.with_ort1
def test_parameter_synchronization(device = rank):
    dp_ways = 1
    num_experts = dist.get_world_size(dist.group.WORLD)
    while dp_ways <= dist.get_world_size():
        # initialize the grid
        ep_ways = dist.get_world_size() // dp_ways
        rank_list = list(range(0, dist.get_world_size()))

        dgrid = DistributionGrid(expert_parallel_group_size = ep_ways, expert_parallel_replica_group_size = dp_ways)

        # build a simple moe layer
        loss = torch.nn.MSELoss()
        num_local_experts = num_experts // ep_ways
        assert num_experts%ep_ways == 0
        d_model = 8

        input = torch.randn(4, 16, d_model).to(device)
        gating_fn = TopKGate(d_model, num_experts, k=2, dgrid=dgrid)  # Top2
        experts = torch.nn.ModuleList()
        for _ in range(num_local_experts):
            expert = torch.nn.Linear(d_model, d_model, bias = False)
            experts.append(expert)

        # randomize weight
        expert.weight = torch.nn.Parameter(torch.randn(d_model, d_model))
        moe = MixtureOfExpertsFunc(gating_fn, experts, distribution_grid= dgrid).to(device)
        if use_ort:
            moe = ORTModule(moe)

        broadcast_parameters(moe, dgrid)

        #gradient checking: pull out the parameters
        dp_params = []
        global_params = []
        dp_params_pre = []
        global_params_pre = []
        for param in moe.parameters():
            if not is_moe_parameter(param):
                global_params_pre.append(param.clone().detach())
                global_params.append(param.data)
            else:
                dp_params.append(param.data)
                dp_params_pre.append(param.clone().detach())

        #1. expert parameters throughout dp_group should be same
        if dp_ways >= 2:
            for param, param_exp in zip(dp_params, dp_params_pre):
                dist.all_reduce(param, op = dist.ReduceOp.MAX, group = dgrid.get_expert_replica_group())
                assert torch.allclose(param_exp, param)

        #2. non expert parameters throughout world should be same
        for param, param_exp in zip(global_params, global_params_pre):
            dist.all_reduce(param, op = dist.ReduceOp.MAX, group = torch.distributed.group.WORLD)
            assert torch.allclose(param_exp, param)

        dp_ways *= 2

# Test single node pipeline parallelism
@pytest.mark.skip(reason="different results between ep_model and pp_model.")
@pytest.mark.with_ort
def test_pipeline_parallelism_single_node(device=rank):
    assert dist.get_world_size() >= 2, "Need at least 2 gpus to test test_pipeline_parallelism_single_node"

    #parameters
    d_model = 4
    d_ff = 8
    num_layers = dist.get_world_size()
    num_experts = dist.get_world_size()
    input = torch.rand(4 * num_experts, 8, d_model).to(device)

    # ep model
    ep_dgrid = DistributionGrid(expert_parallel_group_size = dist.get_world_size())
    ep_gating_fn = TopKGate(d_model, num_experts, k=1, dgrid=ep_dgrid)
    ep_expert = FFNExpert(d_model, d_ff, dgrid=ep_dgrid)
    ep_experts = torch.nn.ModuleList()
    ep_experts.append(ep_expert)
    ep_layers = []
    for i in range(num_layers):
        the_layer = MixtureOfExpertsFunc(ep_gating_fn, ep_experts, distribution_grid=ep_dgrid)
        ep_layers.append(the_layer)
    ep_model = torch.nn.Sequential(*ep_layers).to(device)
    if use_ort:
        ep_model = ORTModule(ep_model)
    ep_output = ep_model(input)

    # pp model
    pp_dgrid = DistributionGrid(num_of_nodes_in_pipeline = 1, num_of_pipeline_stage = dist.get_world_size())
    pp_gating_fn = TopKGate(d_model, num_experts, k=1, dgrid=pp_dgrid)
    pp_expert = FFNExpert(d_model, d_ff, dgrid=pp_dgrid)
    pp_experts = torch.nn.ModuleList()
    pp_experts.append(pp_expert)
    pp_layers = []
    for i in range(num_layers):
        the_layer = MixtureOfExpertsFunc(pp_gating_fn, pp_experts, distribution_grid=ep_dgrid).to(i)
        pp_layers.append(the_layer)

    from torch.distributed.pipeline.sync import Pipe
    pp_model = Pipe(torch.nn.Sequential(*ep_layers), chunks = 2)
    if use_ort:
        pp_model = ORTModule(pp_model)
    pp_output = pp_model(input)

    assert torch.allclose(ep_output, pp_output.local_value())

    #check backward
    loss = torch.nn.MSELoss()
    ep_output = loss(ep_output, input)
    ep_output.backward()
    pp_output = loss(pp_output, input)
    pp_output.backward()
    for p0, p1 in zip(ep_model.parameters(), pp_model.parameters()):
        assert torch.allclose(p0.grad, p1.grad)

@pytest.mark.with_ort1
def test_assertion():
    dgrid = DistributionGrid()
    model_dim = 8
    ff_dim = 12

    num_local_experts = 4
    num_experts = dist.get_world_size(dist.group.WORLD) * num_local_experts

    gating_fn = TopKGate(model_dim, num_experts, k=1, dgrid=dgrid)  # Top1
    experts = torch.nn.ModuleList()
    for i in range(num_local_experts):
        experts.append(FFNExpert(model_dim, ff_dim, dgrid=dgrid)).to("cuda")

    #Test the forward should not be called directly from MixtureOfExperts
    with pytest.raises(NotImplementedError):
        me = MixtureOfExperts(gating_fn, experts, distribution_grid = dgrid).to("cuda")
        if use_ort:
            me = ORTModule(me)
        input = torch.rand(4, 16, model_dim).to("cuda")
        output = me(input)
        if use_ort:
            loss = output[0].sum()
            loss.backward()


    #Test the mergedFFN is not supported in ExpertSlicing
    merged_experts = MergedFFNExpert(model_dim, ff_dim, num_experts, dgrid=dgrid)
    with pytest.raises(AssertionError):
        dgrid = DistributionGrid(expert_slicing_group_size = 1)
        me2 = MixtureOfExpertsFunc(gating_fn, merged_experts, distribution_grid=dgrid).to("cuda")
        if use_ort:
            me2 = ORTModule(me2)
        input = torch.rand(4, 16, model_dim).to("cuda")
        output = me2(input)
        if use_ort:
            loss = output[0].sum()
            loss.backward()

@pytest.mark.with_ort1
def test_fsdp_ep():
    d_model = 512
    nhead = 8
    dim_feedforward = 256
    num_local_experts = 4
    nexperts = dist.get_world_size(dist.group.WORLD) * num_local_experts

    options = { "fsdp_zero_optimization" : {"stage": 1, "flatten_parameters" : False},
                "imbalanced_input_support" : {"enabled" : False}}
    orig_options = {"imbalanced_input_support" : {"enabled" : False}}

    # Original model
    orig_dg = DistributionGrid()
    orig_gate = TopKGate(d_model, nexperts, k=1, dgrid=orig_dg)
    orig_enc = TransformerMoEEncoderLayer(d_model, nhead, dim_feedforward=dim_feedforward,
                                         nexperts=nexperts, gate=orig_gate, distribution_grid = orig_dg,
                                         options = orig_options)
    if use_ort:
        orig_enc = ORTModule(orig_enc)

    orig_params = sum(p.numel() for p in orig_enc.parameters())
    orig_moe_params = 0
    for p in orig_enc.parameters():
        if is_moe_parameter(p):
            orig_moe_params += p.numel()

    # Standard EP distribution
    ep_dg = DistributionGrid(expert_parallel_group_size = dist.get_world_size())
    ep_gate = TopKGate(d_model, nexperts, k=1, dgrid=ep_dg)
    ep_enc = TransformerMoEEncoderLayer(d_model, nhead, dim_feedforward=dim_feedforward,
                                         nexperts=nexperts, gate=ep_gate, distribution_grid = ep_dg,
                                         options = orig_options)
    if use_ort:
        ep_enc = ORTModule(ep_enc)

    ep_params = sum(p.numel() for p in ep_enc.parameters())
    ep_moe_params = 0
    for p in ep_enc.parameters():
        if is_moe_parameter(p):
            ep_moe_params += p.numel()

    # FSDP distribution
    fsdp_dg = DistributionGrid(expert_parallel_group_size = dist.get_world_size())
    fsdp_gate = TopKGate(d_model, nexperts, k=1, dgrid=fsdp_dg)
    fsdp_enc = TransformerMoEEncoderLayer(d_model, nhead, dim_feedforward=dim_feedforward,
                                         nexperts=nexperts, gate=fsdp_gate, distribution_grid = fsdp_dg,
                                         options = options)
    if use_ort:
        fsdp_enc = ORTModule(fsdp_enc)
    fsdp_params = dict(flatten_parameters=False)
    fsdp_enc = fsdp_wrap(fsdp_enc, **fsdp_params)
    fsdp_params = sum(p.numel() for p in fsdp_enc.parameters())
    fsdp_moe_params = 0
    for p in fsdp_enc.parameters():
        if is_moe_parameter(p):
            fsdp_moe_params += p.numel()

    gate_parameters = d_model * nexperts  # number of parameters in gate
    expert_parameters = num_local_experts * 2 * d_model * dim_feedforward  # number of parameters in Experts
    non_expert_parameters = 4 * (d_model * d_model + d_model) # number of attention and the W_o
    non_expert_parameters += 4 * d_model  # number of layernorms

    assert orig_moe_params == size * expert_parameters
    assert orig_params == gate_parameters + size * expert_parameters + non_expert_parameters

    assert ep_moe_params == expert_parameters
    assert ep_params == expert_parameters + non_expert_parameters + gate_parameters

    assert fsdp_moe_params == expert_parameters
    assert fsdp_params == ((non_expert_parameters / size) + gate_parameters + expert_parameters)
    

@pytest.mark.with_ort
def test_einsum():
    # test rules
    M, N, D = 4, 16, 32
    a = torch.rand(M)
    b = torch.rand(M, N)
    rule = 's,se->se'
    assert torch.allclose(om_einsum(rule, a, b), torch.einsum(rule, a, b))
    a = torch.rand(M, N)
    rule = 'se,sc->sec'
    assert torch.allclose(om_einsum(rule, a, b), torch.einsum(rule, a, b))
    rule = 'se,sc->sc'
    assert torch.allclose(om_einsum(rule, a, b), torch.einsum(rule, a, b))
    a = torch.rand(M, N, D)
    rule = 'sec,sm->ecm'
    assert torch.allclose(om_einsum(rule, a, b), torch.einsum(rule, a, b))

def test_enable_zero_optimization_z0():
    options_z0 = { "deepspeed_zero_optimization" : {"stage": 0}}

    dgrid = DistributionGrid()
    model_dim = 8
    ff_dim = 12
    num_experts = 4
    gating_fn = TopKGate(model_dim, num_experts, k=1, dgrid=dgrid)
    experts = torch.nn.ModuleList()
    experts.append(FFNExpert(model_dim, ff_dim, dgrid=dgrid))

    m_z0 = MixtureOfExpertsFunc(gating_fn, experts, distribution_grid=dgrid, options = options_z0)
    for param in m_z0.parameters():
        if is_moe_parameter(param):
            assert hasattr(param, "allreduce") is False
            assert hasattr(param, "group_name") is False

    enc = TransformerMoEEncoderLayer(256, 4, nexperts=8, distribution_grid = dgrid, options = options_z0)
    for param in enc.parameters():
        if is_moe_parameter(param):
            assert hasattr(param, "allreduce") is False
            assert hasattr(param, "group_name") is False

    dec = TransformerMoEDecoderLayer(256, 4, nexperts=8, distribution_grid = dgrid, options = options_z0)
    for param in dec.parameters():
        if is_moe_parameter(param):
            assert hasattr(param, "allreduce") is False
            assert hasattr(param, "group_name") is False

    lenc = LanguageExpertMoEEncoderLayer(256, 4, nexperts=8, distribution_grid = dgrid, options = options_z0)
    for param in lenc.parameters():
        if is_moe_parameter(param):
            assert hasattr(param, "allreduce") is False
            assert hasattr(param, "group_name") is False

    ldec = LanguageExpertMoEDecoderLayer(256, 4, nexperts=8, distribution_grid = dgrid, options = options_z0)
    for param in ldec.parameters():
        if is_moe_parameter(param):
            assert hasattr(param, "allreduce") is False
            assert hasattr(param, "group_name") is False

def test_enable_zero_optimization_z1():
    options_z1 = { "deepspeed_zero_optimization" : {"stage": 1}}

    dgrid = DistributionGrid()
    model_dim = 8
    ff_dim = 12
    num_experts = 4
    gating_fn = TopKGate(model_dim, num_experts, k=1, dgrid=dgrid)
    experts = torch.nn.ModuleList()
    experts.append(FFNExpert(model_dim, ff_dim, dgrid=dgrid))

    m_z1 = MixtureOfExpertsFunc(gating_fn, experts, distribution_grid=dgrid, options = options_z1)
    for param in m_z1.parameters():
        if is_moe_parameter(param):
            assert hasattr(param, "allreduce") is True
            assert param.allreduce == False
            assert hasattr(param, "group_name") is True

    enc = TransformerMoEEncoderLayer(256, 4, nexperts=8, distribution_grid = dgrid, options = options_z1)
    for param in enc.parameters():
        if is_moe_parameter(param):
            assert hasattr(param, "allreduce") is True
            assert param.allreduce == False
            assert hasattr(param, "group_name") is True

    dec = TransformerMoEDecoderLayer(256, 4, nexperts=8, distribution_grid = dgrid, options = options_z1)
    for param in dec.parameters():
        if is_moe_parameter(param):
            assert hasattr(param, "allreduce") is True
            assert param.allreduce == False
            assert hasattr(param, "group_name") is True

    lenc = LanguageExpertMoEEncoderLayer(256, 4, nexperts=8, distribution_grid = dgrid, options = options_z1)
    for param in lenc.parameters():
        if is_moe_parameter(param):
            assert hasattr(param, "allreduce") is True
            assert param.allreduce == False
            assert hasattr(param, "group_name") is True

    ldec = LanguageExpertMoEDecoderLayer(256, 4, nexperts=8, distribution_grid = dgrid, options = options_z1)
    for param in ldec.parameters():
        if is_moe_parameter(param):
            assert hasattr(param, "allreduce") is True
            assert param.allreduce == False
            assert hasattr(param, "group_name") is True

def test_enable_zero_optimization_z2():
    options_z2 = { "deepspeed_zero_optimization" : {"stage": 2}}

    dgrid = DistributionGrid()
    model_dim = 8
    ff_dim = 12
    num_experts = 4
    gating_fn = TopKGate(model_dim, num_experts, k=1, dgrid=dgrid)
    experts = torch.nn.ModuleList()
    experts.append(FFNExpert(model_dim, ff_dim, dgrid=dgrid))

    m_z2 = MixtureOfExpertsFunc(gating_fn, experts, distribution_grid=dgrid, options = options_z2)
    for param in m_z2.parameters():
        if is_moe_parameter(param):
            assert hasattr(param, "allreduce") is True
            assert param.allreduce == False
            assert hasattr(param, "group_name") is True

    enc = TransformerMoEEncoderLayer(256, 4, nexperts=8, distribution_grid = dgrid, options = options_z2)
    for param in enc.parameters():
        if is_moe_parameter(param):
            assert hasattr(param, "allreduce") is True
            assert param.allreduce == False
            assert hasattr(param, "group_name") is True

    dec = TransformerMoEDecoderLayer(256, 4, nexperts=8, distribution_grid = dgrid, options = options_z2)
    for param in dec.parameters():
        if is_moe_parameter(param):
            assert hasattr(param, "allreduce") is True
            assert param.allreduce == False
            assert hasattr(param, "group_name") is True

    lenc = LanguageExpertMoEEncoderLayer(256, 4, nexperts=8, distribution_grid = dgrid, options = options_z2)
    for param in lenc.parameters():
        if is_moe_parameter(param):
            assert hasattr(param, "allreduce") is True
            assert param.allreduce == False
            assert hasattr(param, "group_name") is True

    ldec = LanguageExpertMoEDecoderLayer(256, 4, nexperts=8, distribution_grid = dgrid, options = options_z2)
    for param in ldec.parameters():
        if is_moe_parameter(param):
            assert hasattr(param, "allreduce") is True
            assert param.allreduce == False
            assert hasattr(param, "group_name") is True

def test_enable_zero_optimization_z3():
    options_z3 = { "deepspeed_zero_optimization" : {"stage": 3}}

    dgrid = DistributionGrid()
    model_dim = 8
    ff_dim = 12
    num_experts = 4
    gating_fn = TopKGate(model_dim, num_experts, k=1, dgrid=dgrid)
    experts = torch.nn.ModuleList()
    experts.append(FFNExpert(model_dim, ff_dim, dgrid=dgrid))

    m_z3 = MixtureOfExpertsFunc(gating_fn, experts, distribution_grid=dgrid, options = options_z3)
    for param in m_z3.parameters():
        if is_moe_parameter(param):
            assert hasattr(param, "allreduce") is True
            assert param.allreduce == False
            assert hasattr(param, "group_name") is True

    enc = TransformerMoEEncoderLayer(256, 4, nexperts=8, distribution_grid = dgrid, options = options_z3)
    for param in enc.parameters():
        if is_moe_parameter(param):
            assert hasattr(param, "allreduce") is True
            assert param.allreduce == False
            assert hasattr(param, "group_name") is True

    dec = TransformerMoEDecoderLayer(256, 4, nexperts=8, distribution_grid = dgrid, options = options_z3)
    for param in dec.parameters():
        if is_moe_parameter(param):
            assert hasattr(param, "allreduce") is True
            assert param.allreduce == False
            assert hasattr(param, "group_name") is True

    lenc = LanguageExpertMoEEncoderLayer(256, 4, nexperts=8, distribution_grid = dgrid, options = options_z3)
    for param in lenc.parameters():
        if is_moe_parameter(param):
            assert hasattr(param, "allreduce") is True
            assert param.allreduce == False
            assert hasattr(param, "group_name") is True

    ldec = LanguageExpertMoEDecoderLayer(256, 4, nexperts=8, distribution_grid = dgrid, options = options_z3)
    for param in ldec.parameters():
        if is_moe_parameter(param):
            assert hasattr(param, "allreduce") is True
            assert param.allreduce == False
            assert hasattr(param, "group_name") is True

def test_enable_expert_weight_calculation_optimization(device=rank):
    model_dim = 8
    num_local_experts = 4
    num_experts = dist.get_world_size(dist.group.WORLD) * num_local_experts
    input = torch.rand(8, 16, model_dim).to(device)
    dgrid = DistributionGrid(expert_parallel_group_size = dist.get_world_size())
    gate = TopKGate(model_dim, num_experts, k=2, dgrid=dgrid)
    experts = torch.nn.ModuleList()
    for i in range(num_local_experts):
        expert = torch.nn.Linear(model_dim, model_dim, bias=False)
        # use identify matrix
        expert.weight = torch.nn.Parameter(torch.eye(model_dim))
        experts.append(expert)
    options = {"enable_expert_weight_calculation_optimization" : True}
    moe = MixtureOfExpertsFunc(gate, experts, distribution_grid = dgrid, options = options).to(device)
    output = moe(input)
    assert output.shape == input.shape
