# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import copy
import random
import torch
import torch.distributed as dist
from torch._C import default_generator

from .moe import MixtureOfExperts
from .topKgate import TopKGate

_expert_state_dict_keyword = "moe_experts"

def get_expert_parameters_list(model):
    """Returns list of expert parameters"""
    return [p for p in model.parameters() if is_moe_parameter(p)]

def get_expert_parameters_state_dict(model):
    """
    Returns state dict of expert parameters

    Warning:
    Since torch.nn.Module.state_dict() returns detached (shallow) copies
    of parameters, so does this function.
    Among other this, that means that `is_moe_parameter` returns False
    for the values in this dictionary.
    """
    return {
        k: v for k, v in model.state_dict().items() if _expert_state_dict_keyword in k
    }

def get_non_expert_parameters_list(model):
    """Returns list of non-expert parameters"""
    return [p for p in model.parameters() if not is_moe_parameter(p)]

def get_non_expert_parameters_state_dict(model):
    """Returns state dict of non-expert parameters"""
    return {
        k: v
        for k, v in model.state_dict().items()
        if _expert_state_dict_keyword not in k
    }

def get_state_dict_for_local_expert_idx(model, local_expert_idx):
    """
    Returns state dict of expert parameters for a specific local expert.

    Warning:
    This function currently does not support MergedFFNExpert.

    Warning:
    Since torch.nn.Module.state_dict() returns detached (shallow) copies
    of parameters, so does this function.
    Among other this, that means that `is_moe_parameter` returns False
    for the values in this dictionary.
    """
    for module in model.modules():
        if isinstance(module, MixtureOfExperts) and module.is_mergedFFNExpert:
            raise NotImplementedError("This function does not support MergedFFNExpert.")

    keyword = f"{_expert_state_dict_keyword}.{local_expert_idx}."
    return {k: v for k, v in model.state_dict().items() if keyword in k}

def get_state_dict_partitions_for_saving(model, dgrid, total_num_experts):
    """
    Returns a (distributed) partition of the state dict of the given MoE model for saving checkpoints.

    Warning:
    This function currently does not support MergedFFNExpert.

    This function can be used in conjunction with get_state_dict_partition_names_for_loading
    for efficient checkpointing of MoE models.
    These functions even work if you change the number of GPUs
    between saving and loading (as long as the model fits into GPU, of course).
    For example, you could save a model trained on 16 GPUs
    and then load it on 8 GPUs for inference.

    To understand what this function does, it's best to look at an example.
    Assume you have a model with 8 experts without expert replicas on 4 GPUs.
    To create a complete checkpoint of the model, you should call this function on every rank.
    The function will return a dictionary on each rank.
    The keys of these dictionaries would be:

    rank 0: ["skeleton", "expert0", "expert1"]
    rank 1: ["expert2", "expert3"]
    rank 2: ["expert4", "expert5"]
    rank 3: ["expert6", "expert7"]

    The values corresponding to these keys are again dictionaries.
    The values form a partition of the state dict of the entire model:
    The value corresponding to the key "skeleton" is a dictionary that contains all
    parameters in the non-expert part of the model.
    The value corresponding to the key "expert<k>" is a dictionary that contains the
    parameters of all experts (in all MoE layers) with global index k.

    To save a checkpoint of the entire model,
    you can simple dump all partitions into separate files on all GPUs:
    ```
    partitions = get_state_dict_partitions_for_saving(model, grid, total_num_experts)
    for name, state_dict in partitions.items():
        save_path = os.path.join(save_dir, f"{name}.pt")
        torch.save(state_dict, save_path)
    ```
    """
    for module in model.modules():
        if isinstance(module, MixtureOfExperts) and module.is_mergedFFNExpert:
            raise NotImplementedError("This function does not support MergedFFNExpert.")

    partitions = {}

    # global rank 0 saves skeleton (non-expert) parameters
    if dist.get_rank() == 0:
        partitions["skeleton"] = get_non_expert_parameters_state_dict(model)

    # every node with expert parallel replica rank 0 saves its experts
    if dgrid.get_expert_parallel_replica_rank() == 0:
        local_expert_idx = 0
        while True:
            local_state_dict = get_state_dict_for_local_expert_idx(
                model, local_expert_idx
            )
            if not local_state_dict:  # empty
                break  # done
            global_expert_idx = dgrid.map_expert_id_local_to_global(total_num_experts, local_expert_idx)
            if isinstance(global_expert_idx, tuple):
                global_expert_idx = global_expert_idx[0]
            replace_from = f"{_expert_state_dict_keyword}.{local_expert_idx}"
            replace_to = f"{_expert_state_dict_keyword}.{global_expert_idx}"
            global_state_dict = {
                k.replace(replace_from, replace_to): v
                for k, v in local_state_dict.items()
            }
            partitions[f"expert{global_expert_idx}"] = global_state_dict
            local_expert_idx += 1

    return partitions

def get_state_dict_partition_names_for_loading(model, dgrid, total_num_experts):
    """
    Returns a list of partition names for loading a checkpoint.

    Warning:
    This function currently does not support MergedFFNExpert.

    This function can be used in conjunction with get_state_dict_partitions_for_saving
    for efficient checkpointing of MoE models.
    These functions even work if you change the number of GPUs
    between saving and loading (as long as the model fits into GPU, of course).
    For example, you could save a model trained on 16 GPUs
    and then load it on 8 GPUs for inference.

    Please read the documentation of get_state_dict_partitions_for_saving
    before proceeding.
    This function returns the names of the partitions a GPU has to load
    in order to load an MoE model.
    For example, if you load a model with 8 experts without expert replicas on 4 GPUs,
    this function would return the following lists:

    rank 0: ["skeleton", "expert0", "expert1"]
    rank 1: ["skeleton", "expert2", "expert3"]
    rank 2: ["skeleton", "expert4", "expert5"]
    rank 3: ["skeleton", "expert6", "expert7"]

    Loading a checkpoint that has been saved with get_state_dict_partitions_for_saving looks like this:
    ```
    partition_names = get_state_dict_partition_names_for_loading(model, grid, num_experts)
    state_dict = {}
    for name in partition_names:
        load_path = os.path.join(load_dir, f"{name}.pt")
        global_state_dict = torch.load(load_path, map_location=f"cuda:{local_rank}")
        local_state_dict = translate_state_dict_global_to_local(global_state_dict, grid, num_experts)
        state_dict.update(local_state_dict)
    model.load_state_dict(state_dict)
    ```
    The function `translate_state_dict_global_to_local` is defined below.
    """
    partition_names = ["skeleton"]
    for global_expert_id in range(total_num_experts):
        mapped_expert_id = dgrid.map_expert_id_global_to_local(total_num_experts, global_expert_id)
        if isinstance(mapped_expert_id, list):
            relevant_ranks = [x[0] for x in mapped_expert_id]
        else:
            relevant_ranks = [mapped_expert_id[0]]
        if dist.get_rank() in relevant_ranks:
            partition_names.append(f"expert{global_expert_id}")
    return partition_names

def _translate_state_dict_key(key, dgrid, total_num_experts, local_to_global):
    """
    Translates between local and global expert indices in state dict keys.

    Note: There are more user-friendly functions for this purpose below.

    Translates from local to global indices if local_to_global=True
    and from global to local, otherwise.
    If the key belongs to a non-expert parameter
    (i.e., it does not contain the string "moe_experts")
    the function returns the given key unchanged.

    Example:
    Let's assume you have a model with 8 experts and without expert replicas on 2 GPUs.
    If you call this on rank 1 with local_to_global=True and
    key="decoder.decoders.1.feed_forward.moe_experts.2.linear2.weight",
    this function would return
    "decoder.decoders.1.feed_forward.moe_experts.5.linear2.weight"
    """
    # extract expert index from key
    keyword_start_idx = key.find(_expert_state_dict_keyword)
    if keyword_start_idx == -1:  # non-expert parameter, no translation needed
        return key
    index_start_idx = keyword_start_idx + len(_expert_state_dict_keyword) + 1
    index_len = key[index_start_idx:].find(".")
    assert index_len >= 0, "invalid key"
    expert_idx = int(key[index_start_idx : index_start_idx + index_len])

    # translate expert index
    if local_to_global:
        translated_expert_idx = dgrid.map_expert_id_local_to_global(total_num_experts, expert_idx)
        if isinstance(translated_expert_idx, tuple):
            translated_expert_idx = translated_expert_idx[0]
    else:
        mapped_expert_id = dgrid.map_expert_id_global_to_local(total_num_experts, expert_idx)
        if isinstance(mapped_expert_id, list):
            d = {rank: idx for (rank, idx) in mapped_expert_id}
            translated_expert_idx = d[dist.get_rank()]
        else:
            rank, idx = mapped_expert_id
            assert rank == dist.get_rank()
            translated_expert_idx = idx

    # construct translated key
    return (
        key[:index_start_idx]
        + str(translated_expert_idx)
        + key[index_start_idx + index_len :]
    )

def translate_state_dict_key_local_to_global(local_key, dgrid, total_num_experts):
    """
    Translates a local expert index to a global expert index in a state dict key.

    If the key belongs to a non-expert parameter
    (i.e., it does not contain the string "moe_experts")
    the function returns the given key unchanged.

    Example:
    Let's assume you have a model with 8 experts and without expert replicas on 2 GPUs.
    If you call this on rank 1 with
    key="decoder.decoders.1.feed_forward.moe_experts.2.linear2.weight",
    this function would return
    "decoder.decoders.1.feed_forward.moe_experts.5.linear2.weight"
    """
    return _translate_state_dict_key(local_key, dgrid, total_num_experts, True)

def translate_state_dict_key_global_to_local(global_key, dgrid, total_num_experts):
    """
    Translates a global expert index to a local expert index in a state dict key.

    If the key belongs to a non-expert parameter
    (i.e., it does not contain the string "moe_experts")
    the function returns the given key unchanged.

    Example:
    Let's assume you have a model with 8 experts and without expert replicas on 2 GPUs.
    If you call this on rank 1 with
    key="decoder.decoders.1.feed_forward.moe_experts.5.linear2.weight",
    this function would return
    "decoder.decoders.1.feed_forward.moe_experts.2.linear2.weight"
    """
    return _translate_state_dict_key(global_key, dgrid, total_num_experts, False)

def translate_state_dict_local_to_global(local_state_dict, dgrid, total_num_experts):
    """
    Translates a state dict to use global expert indices instead of local expert indices.
    """
    return {
        translate_state_dict_key_local_to_global(k, dgrid, total_num_experts): v
        for k, v in local_state_dict.items()
    }

def translate_state_dict_global_to_local(global_state_dict, dgrid, total_num_experts):
    """
    Translates a state dict to use local expert indices instead of global expert indices.
    """
    return {
        translate_state_dict_key_global_to_local(k, dgrid, total_num_experts): v
        for k, v in global_state_dict.items()
    }

'''
Exclude the expert tensors from DDP. Only works for torch.nn.parallel.DistributedDataParallel
'''
def exclude_moe_params_in_ddp(model, moe_name = "expert"):
    assert torch.distributed.is_initialized(), "exclude_moe_params_in_ddp only supports Torch DDP"
    model._ddp_params_and_buffers_to_ignore = []
    for p in model.named_parameters():
        if moe_name in p[0]:
            model._ddp_params_and_buffers_to_ignore.append(p[0])

def moe_module_all_reduce_non_experts(model, dgrid):
    assert dist.is_initialized(), "MoE allreduce tensor update only support Torch DDP"

    #For non moe related parameters, accumulate cross the world
    pg = dgrid.get_data_parallel_group()
    pg_size = dgrid.get_data_parallel_world_size()
    if pg is None:
        pg = dist.group.WORLD
        pg_size = dist.get_world_size()

    for param in get_non_expert_parameters_list(model):
        if param.grad is None:
            # In cases where there is an imbalance of empty grads across
            # ranks we must create empty grads, this will ensure that every
            # rank is reducing the same size. In some cases it may make
            # sense in the future to support the ability to average not
            # w.r.t. world size but with a different value.
            param.grad = torch.zeros(param.size(),
                                        dtype=param.dtype,
                                        device=param.device)
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, group=pg)
        param.grad.data /= pg_size

def moe_module_all_reduce_experts(model, dgrid):
    assert dist.is_initialized(), "MoE allreduce tensor update only support Torch DDP"
    #For moe related parameters, accumulate cross the dp groups
    # no need to iterate over all dp groups, just pass the native dp_group
    local_dp_group = dgrid.get_expert_replica_group()
    if local_dp_group is None:
        return

    for param in get_expert_parameters_list(model):
        if param.grad is None:
            # In cases where there is an imbalance of empty grads across
            # ranks we must create empty grads, this will ensure that every
            # rank is reducing the same size. In some cases it may make
            # sense in the future to support the ability to average not
            # w.r.t. world size but with a different value.
            param.grad = torch.zeros(param.size(),
                                            dtype=param.dtype,
                                            device=param.device)
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, group = local_dp_group)
        param.grad.data /= dist.get_world_size(local_dp_group)

def moe_module_all_reduce(model, dgrid):
    assert dist.is_initialized(), "MoE allreduce tensor update only support Torch DDP"
    moe_module_all_reduce_non_experts(model, dgrid)
    moe_module_all_reduce_experts(model, dgrid)

'''
Override the scale_check_overflow_python and axpby_check_overflow_python in apex.amp.scaler to add in the
allreduce to check and propagate overflow flag.
'''
def apex_amp_scale_check_overflow_override(amp):
    from mpi4py import MPI
    def is_using_mpi():
        return(MPI.COMM_WORLD.Get_size() == torch.distributed.get_world_size(torch.distributed.group.WORLD))
    def update_scale(self):
        # If the fused kernel is available, we only need one D2H memcopy and sync.
        if self.has_fused_kernel and self.dynamic and not self._has_overflow:
            self._has_overflow = self._overflow_buf.item()

        should_skip = self._has_overflow and self.dynamic

        if torch.distributed.is_initialized():
            if is_using_mpi():
                should_skip = MPI.COMM_WORLD.allreduce(should_skip, MPI.MAX)
            else:
                local_rank = dist.get_rank() %  torch.cuda.device_count() #TODO: is there a better way to get local rank?
                should_skip = torch.tensor(should_skip).to(local_rank)
                dist.all_reduce(should_skip, op=dist.ReduceOp.MAX)
                should_skip = should_skip.item()

        if should_skip:
            if(self._min_loss_scale):
                self._loss_scale = max(self._min_loss_scale, self._loss_scale/2.)
            else:
                self._loss_scale = self._loss_scale/2.
            self._unskipped = 0
        else:
            self._unskipped += 1

        if self._unskipped == self._scale_seq_len and self.dynamic:
            self._loss_scale = min(self._max_loss_scale, self._loss_scale*2.)
            self._unskipped = 0

        return should_skip

    amp.scaler.LossScaler.update_scale = update_scale

def is_moe_parameter(p):
    return hasattr(p, 'is_moe_param') and p.is_moe_param

def contain_only_moe_parameters(module):
    """
    Check if the module contains moe parameters only

    Note: Return False if the module contains no parameters
    """
    is_empty = True
    for parameter in module.parameters():
        is_empty = False
        if not is_moe_parameter(parameter):
            return False
    return True if not is_empty else False

def is_gate_parameter(p):
    return hasattr(p, 'is_gate_param') and p.is_gate_param

def contain_only_gate_parameters(module):
    """
    Check if the module contains gate parameters only

    Note: Return False if the module contains no parameters
    """
    is_empty = True
    for parameter in module.parameters():
        is_empty = False
        if not is_gate_parameter(parameter):
            return False
    return True if not is_empty else False

def get_num_parameters(module):
    """
    Compute number of parameters in a module that uses MoE.
    """
    def count_parameters(module, condition=(lambda p:True)):
        return sum(p.numel() for p in module.parameters() if p.requires_grad and condition(p))

    per_gpu = count_parameters(module)
    in_experts_per_gpu = count_parameters(module, condition=is_moe_parameter)

    if dist.is_initialized():
        total = per_gpu + in_experts_per_gpu * (dist.get_world_size() - 1)
    else:
        total = per_gpu

    return per_gpu, in_experts_per_gpu, total

class TemporaryRngState:
    '''
    Context manager for working with a temporary random number generator (RNG) state.

    The constructor gets a random number from the Python RNG that is used as
    (part of) the seed for the temporary RNG
    and then stores the current RNG state to restore the it later on.
    If add_rank_to_seed=True, the GPU rank is added to the seed.
    This is useful to initialize MoE models
    where the experts on different GPUs should be initialized independently.
    Note that this feature requires torch.distributed to be initialized.

    On enter, the context managers sets the RNG state to the random seed created in the constructor
    to establish a temporary RNG state.

    On exit, the context manager resets the RNG state to the previously remembered state.
    Thereby, any RNG operations executed with this context manager
    do not affect the global, non-temporary RNG state.
    However, the usage of this context manager does advance the Python RNG
    since it uses that RNG to generate the random seed in the constructor.

    The context manager resets the Python RNG state and
    the PyTorch RNG state for CPU and GPU (if cuda is initialized).
    It does not currently reset the numpy RNG state.
    '''
    def __init__(self, add_rank_to_seed=False):
        self.seed = random.randrange(2**32)
        if add_rank_to_seed:
            assert dist.is_initialized()
            self.seed += dist.get_rank()
        self.python_rng_state = random.getstate()
        self.torch_rng_state = torch.get_rng_state()
        if torch.cuda.is_initialized():
            self.torch_rng_state_cuda = torch.cuda.get_rng_state()

    def __enter__(self):
        # increment seed for different RNGs to avoid correlation
        # in the (very unlikely) case that the different RNGs
        # use the exact same algorithm
        random.seed(self.seed)
        # do not call torch.maunal_seed here, because that sets the seed of all GPUs
        default_generator.manual_seed(self.seed + 1)
        if torch.cuda.is_initialized():
            torch.cuda.manual_seed(self.seed + 2)  # only set seed of default cuda device

    def __exit__(self, exc_type, exc_value, exc_traceback):
        random.setstate(self.python_rng_state)
        torch.set_rng_state(self.torch_rng_state)
        if torch.cuda.is_initialized():
            torch.cuda.set_rng_state(self.torch_rng_state_cuda)

class Synchronizer:
    """
    Context manager for synchronizing forward-passes on the given module across GPUs
    to make sure that all GPUs execute the same number of forward-passes.
    If the code within the context manager executes fewer forward-passes on a certain GPU,
    the context manager will make sure that the GPU keeps doing forward-passes with the
    given dummy_batch until all GPUs exit the context manager.

    This class dynamically import mpi4py.
    """
    def __init__(self, module, dummy_batch, *forward_args, enabled=True, debug=False, dgrid=None, **forward_kwargs):
        self.module = module
        self.dummy_batch = copy.deepcopy(dummy_batch)
        self.enabled = enabled
        self.forward_args = forward_args
        self.forward_kwargs = forward_kwargs
        self.debug = debug
        self.batch_counter = 0

        if dgrid:
                self.mpi_expert_group = dgrid.get_mpi_group_for_expert_group()
        else:
            from mpi4py import MPI
            self.mpi_expert_group = MPI.COMM_WORLD

    def __enter__(self):
        if self.enabled:
            self.hook_handle = self.module.register_forward_pre_hook(self.forward_hook)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.enabled:
            from mpi4py import MPI
            # remove hook so that the function forward_hook is not called for dummy batches
            self.hook_handle.remove()
            # run forward-passes with dummy-batches until all ranks send True,
            # which means that all ranks are done processing their batches
            # this rank sends True to signal that it is done with its batches
            while not self.mpi_expert_group.allreduce(True, MPI.LAND):
                if self.debug:
                    print(f"batch {self.batch_counter}: dummy")
                self.batch_counter += 1
                self.module(copy.deepcopy(self.dummy_batch), *self.forward_args, **self.forward_kwargs)

    def forward_hook(self, module, input):
        from mpi4py import MPI
        if self.debug:
            print(f"batch {self.batch_counter}: real")
        self.batch_counter += 1
        # this rank sends False to signal that there are still batches to be processed
        self.mpi_expert_group.allreduce(False, MPI.LAND)

def broadcast_parameters(model, dgrid):
    """
    A helper function for making sure weights are initialized same across ranks
        It broadcasts non-expert weights from rank 0 to whole world
        and expert weights from lowest rank in expert replica group to all the nodes in replica group
    """
    assert dist.is_initialized(), "Torch distributed is not initialized!"
    expert_replica_group = dgrid.get_expert_replica_group()
    epr_src_rank  = dgrid.get_expert_replica_src_rank()

    # Broadcast non-expert parameters cross the world from rank 0
    for param in get_non_expert_parameters_list(model):
        dist.broadcast(param, 0, group=dist.group.WORLD)

    # Broadcast expert parameters across the dp groups
    if expert_replica_group is not None:
        for param in get_expert_parameters_list(model):
            dist.broadcast(param, epr_src_rank, group=expert_replica_group)

def get_moe_loss(model, layer_level=False, dtype=torch.float32, condition=None):
    moe_loss = None
    gate_log = None
    num_moe_layers = 0
    for name, p in model.named_modules():
        if isinstance(p, MixtureOfExperts) and p.gate.loss is not None and (condition is None or condition in name):
            if num_moe_layers == 0:
                moe_loss = p.gate.loss.clone()
                gate_log = {k: v.detach().clone() for k, v in p.gate.gate_log.items()}
            else:
                moe_loss += p.gate.loss.clone()
                for k, v in p.gate.gate_log.items():
                    if k in gate_log:  # necessary when different layers have different topk gates
                        gate_log[k] += v.detach().clone()
                    else:
                        gate_log[k] = v.detach().clone()
            if layer_level:
                for k, v in p.gate.gate_log.items():  # per-layer log
                    if v.ndim == 0:  # only log gate-level metrics, do not log expert-level metrics
                        assert f"{k}_l{num_moe_layers}" not in gate_log
                        gate_log[f"{k}_l{num_moe_layers}"] = v.detach().clone()
            num_moe_layers += 1
    moe_loss = moe_loss.to(dtype) if moe_loss is not None else None
    return moe_loss, gate_log, num_moe_layers

def clear_moe_loss(model, condition=None):
    for name, p in model.named_modules():
        if isinstance(p, MixtureOfExperts) and p.gate.loss is not None and (condition is None or condition in name):
            p.gate.loss = None

def set_gate_metrics(model, balance_ratio=None, gate_log_req=None):
    for p in model.modules():
        if isinstance(p, TopKGate):
            p.set_gate_metrics(balance_ratio, gate_log_req)

def fsdp_wrap(module, **kwargs):
    """
    Wrap nn.moule in FSDP. If fairscale is not available then return the module as it is.
    """
    from fairscale.nn.wrap import wrap, enable_wrap, auto_wrap
    from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
    enable_wrap_params = kwargs
    if "wrapper_cls" not in kwargs:
        enable_wrap_params = dict(kwargs,wrapper_cls=FSDP)
    with enable_wrap(**enable_wrap_params):
        module = wrap(module, **kwargs)
    return module
