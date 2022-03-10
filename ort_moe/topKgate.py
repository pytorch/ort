# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
#
# The file has been adapted fairscale file:
#   https://github.com/facebookresearch/fairscale/blob/master/fairscale/nn/moe/top2gate.py
# We retain the following license from the original files:

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# Implementation of Top2Gating described in https://arxiv.org/pdf/2006.16668.pdf
# Code is inspired by Top2GatingOnLogits from lingvo:
#   https://github.com/tensorflow/lingvo/blob/21b8106c5f1d30a196c98eedc441d4fd70833b11/lingvo/core/moe_layers.py#L477

from typing import Callable, Dict, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F
import math

from .loss_functions import loss_functions
from .gate_logs import gate_logs
from .custom_ops import einsum

uniform_map: Dict[torch.device, Callable] = {}
gumbel_map: Dict[torch.device, Callable] = {}

def multiplicative_jitter(x, device: torch.device, epsilon=1e-2):
    """
    From Mesh Transformers MoE.

	Multiply values by a random number between 1-epsilon and 1+epsilon.

    Makes models more resilient to rounding errors introduced by bfloat16.
    This seems particularly important for logits.

    Args:
        x: a torch.tensor
        device: torch.device
        epsilon: a floating point value

    Returns:
        a torch.tensor with the same type and shape as x.
    """
    if epsilon == 0:
        return x
    #TODO: Why does torch.tensor(1.0-epsilon, device =device) fail?
    minval = torch.empty([], device = device).fill_(1.0-epsilon)
    maxval = torch.empty([], device = device).fill_(1.0+epsilon)
    uniform = uniform_map.get(device)
    if uniform is None:
        uniform = torch.distributions.uniform.Uniform(low=minval, high=maxval).rsample
        uniform_map[device] = uniform
    return x * uniform(x.shape)

def gumbel_rsample(shape: Tuple, device: torch.device) -> Tensor:
    gumbel = gumbel_map.get(device)
    if gumbel is None:
        one = torch.tensor(1.0, device=device)
        zero = torch.tensor(0.0, device=device)
        gumbel = torch.distributions.gumbel.Gumbel(zero, one).rsample  # type: ignore
        gumbel_map[device] = gumbel
    return gumbel(shape)

def top2gating(logits: torch.Tensor, capacity_factor: float, fp16_mode: bool=False, nonpadding: torch.Tensor=None,
                logits_gumbel: float=0.0, token_drop_type: str='cut', second_place_loss_ratio: float=0.0,
                straight_through: bool=False, straight_through_temperature: float=1.0,
                balance_ratio={'load_balance': 0.01}, gate_log_req: dict={}, lid: int=-1,
                tutel_cumsum_sub_one: callable=None)-> Tuple[Tensor, Tensor, Tensor, Tensor, float]:
    """Implements Top2Gating on logits."""
    if fp16_mode is True:
        logits = logits.to(torch.float32)
    gates = F.softmax(logits, dim=1) #dim: [bs, num_experts]
    if straight_through:
        gates_st = F.softmax(logits/straight_through_temperature, dim=1) if straight_through_temperature != 1.0 else gates

    # gates has shape of SE
    num_tokens = gates.shape[0]
    num_experts = gates.shape[1]
    num_nonpadding = torch.sum(nonpadding) if nonpadding is not None else num_tokens
    # capacity = 2S/E
    capacity_fp = max(min(num_tokens, (2 * num_tokens / num_experts) * capacity_factor), 4)
    capacity = math.ceil(capacity_fp)

    # Create a mask for 1st's expert per token
    indices1_s = torch.argmax(gates, dim=1)  #dim: [bs], the index of the expert with highest softmax value
    mask1 = fast_one_hot(indices1_s, num_classes=num_experts)  #dim: [bs, num_experts]. 1 for the expert with highest softmax value
    if lid >= 0 and (torch.sum(mask1.float(), dim=0).int() == 0).any():
        print(f"WARNING: top2gating: expert got too few top1 examples in layer {lid}: {torch.sum(mask1.float(), dim=0).int().tolist()}")
    # mask using nonpadding (https://github.com/tensorflow/mesh/blob/a54f5cf75ef44d8a97190b3e5aaec176c138b3c0/mesh_tensorflow/transformer/moe.py#L1399)
    gates_nonpadding = gates
    if straight_through:
        gates_st_nonpadding = gates_st
    if nonpadding is not None:
        mask1 = einsum("s,se->se", nonpadding, mask1)
        gates_nonpadding = einsum("s,se->se", nonpadding, gates_nonpadding)
        if straight_through:
            gates_st_nonpadding = einsum("s,se->se", nonpadding, gates_st_nonpadding) if straight_through_temperature != 1.0 else gates_nonpadding

    gates_st_equals_gates = True
    if logits_gumbel > 0:
        # Create a mask for 2nd's expert per token using Gumbel-max trick
        # https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/
        logits_w_noise = logits + logits_gumbel * gumbel_rsample(logits.shape, device=logits.device)
        if straight_through:
            gates_st = F.softmax(logits_w_noise/straight_through_temperature, dim=1)
            gates_st_equals_gates = False

    # Replace top-expert with min value
    gates_without1 = gates * (1.0-mask1)
    if straight_through:
        gates_st_without1 = gates_st * (1.0-mask1) if not gates_st_equals_gates else gates_without1
    logits_except1 = (logits_w_noise if logits_gumbel > 0 else logits).masked_fill(mask1.bool(), float("-inf"))
    indices2_s = torch.argmax(logits_except1, dim=1)  #dim: [bs], the index of the expert with the second highest softmax value
    mask2 = fast_one_hot(indices2_s, num_classes=num_experts)  #dim: [bs, num_experts]. 1 for the expert with the second highest softmax value
    if lid >= 0 and (torch.sum(mask2.float(), dim=0).int() == 0).any():
        print(f"WARNING: top2gating: expert got too few 2nd top examples in layer {lid}: {torch.sum(mask2.float(), dim=0).int().tolist()}")
    # mask using nonpadding (https://github.com/tensorflow/mesh/blob/a54f5cf75ef44d8a97190b3e5aaec176c138b3c0/mesh_tensorflow/transformer/moe.py#L1408)
    gates_without1_nonpadding = gates_without1
    if straight_through:
        gates_st_without1_nonpadding = gates_st_without1
    if nonpadding is not None:
        mask2 = einsum("s,se->se", nonpadding, mask2)
        gates_without1_nonpadding = einsum("s,se->se", nonpadding, gates_without1)
        if straight_through:
            gates_st_without1_nonpadding = einsum("s,se->se", nonpadding, gates_st_without1) if not gates_st_equals_gates else gates_without1_nonpadding

    # Compute l_aux
    # the fraction of the router probability allocated for each expert
    me1 = torch.sum(gates_nonpadding, dim=0) / num_nonpadding
    # the fraction of tokens dispatched to each expert
    ce1 = torch.sum((mask1.float()-gates_st_nonpadding).detach()+gates_st_nonpadding if straight_through else mask1.float(), dim=0) / num_nonpadding
    # Also add a loss to encourage all experts to be used equally also as the
    # second-place expert.  Experimentally, this seems to be a wash.
    # As a proxy for ce2, we renormalize the raw gates after the top one
    # has been removed.
    normalized = gates_without1_nonpadding / (torch.sum(gates_without1, dim=1, keepdim=True) + 1e-9)
    if straight_through:
        normalized_st = gates_st_without1_nonpadding / (torch.sum(gates_st_without1, dim=1, keepdim=True) + 1e-9) if not gates_st_equals_gates else normalized
    me2 = torch.sum(normalized, dim=0) / num_nonpadding
    # We want to equalize the fraction of the batch assigned to each expert:
    ce2 = torch.sum((mask2.float()-normalized_st).detach()+normalized_st if straight_through else mask2.float(), dim=0) / num_nonpadding

    raw_mask1 = mask1.clone().detach()
    raw_mask2 = mask2.clone().detach()

    if token_drop_type in ['random', 'routing_weight']:
        mask12 = torch.max(mask1, mask2)
        if token_drop_type == 'random':
            # randomly select masked tokens to fit in capacity buffer
            priority = mask12 * torch.rand_like(mask1.float())  # dim: [bs, num_experts]
        else:
            priority = mask12 * gates_nonpadding  # dim: [bs, num_experts]
        _, cap_idx = torch.topk(priority, k=min(capacity, priority.shape[0]), dim=0)  # dim: [capacity, num_experts]
        priority_mask = torch.zeros_like(mask1).scatter_(0, cap_idx, 1)
        mask1 *= priority_mask
        mask2 *= priority_mask

    # Compute locations in capacity buffer
    if mask1.device.type == 'cpu' or tutel_cumsum_sub_one is None:
        locations1 = torch.cumsum(mask1, dim=0) - 1
        locations2 = torch.cumsum(mask2, dim=0) - 1
    else:
        locations1 = tutel_cumsum_sub_one(mask1, dim=0)
        locations2 = tutel_cumsum_sub_one(mask2, dim=0)
    # Update 2nd's location by accounting for locations of 1st
    locations2 += torch.sum(mask1, dim=0, keepdim=True)
    if token_drop_type == 'cut':
        # Remove locations outside capacity from mask
        mask1 *= torch.lt(locations1, capacity)
        mask2 *= torch.lt(locations2, capacity)

    # Store the capacity location for each token
    locations1_s = torch.sum(locations1 * mask1, dim=1)
    locations2_s = torch.sum(locations2 * mask2, dim=1)

    # Normalize gate probabilities
    mask1_float = mask1.float()
    mask2_float = mask2.float()

    gates12_s = torch.zeros(2, num_tokens).to(gates_nonpadding) #[K, S], stores the weights of the top2 softmax, for each token

    gates12_s[0] = torch.sum(gates_nonpadding.reshape(gates_nonpadding.shape[0], -1) * mask1_float, dim = 1)
    gates12_s[1] = torch.sum(gates_nonpadding.reshape(gates_nonpadding.shape[0], -1) * mask2_float, dim = 1)

    loss, gate_log = compute_gate_loss(balance_ratio, gate_log_req,
                                        logits=logits, gates=gates_nonpadding, gates_max=gates12_s[0],
                                        raw_mask=raw_mask1, routing_mask=mask1,
                                        router_prob_fraction=me1, token_dispatch_fraction=ce1, nonpadding=nonpadding,
                                        num_experts=num_experts, num_nonpadding=num_nonpadding)
    loss2, gate_log2 = compute_gate_loss(balance_ratio, gate_log_req,
                                        logits=logits, gates=gates_nonpadding, gates_max=gates12_s[1],
                                        raw_mask=raw_mask2, routing_mask=mask2,
                                        router_prob_fraction=me2, token_dispatch_fraction=ce2, nonpadding=nonpadding,
                                        num_experts=num_experts, num_nonpadding=num_nonpadding)
    if second_place_loss_ratio > 0:
        loss += loss2 * second_place_loss_ratio
    for k, v in gate_log2.items():
        gate_log[f'{k}_2nd'] = v

    denom_s = torch.sum(gates12_s, dim=0) + 1e-9
    gates12_s /= denom_s

    # Calculate combine_weights and dispatch_mask
    dispatch_mask = torch.full((num_experts * capacity+1,), -1, dtype=int, device=logits.device)
    indices = torch.arange(0, num_tokens, dtype=int, device=logits.device)
    mask_sum1 = torch.sum(mask1, dim=1)
    #dim [S], each stores the indices in the dispatch_mask the current token resides. E.g, if dispatch_indices1[0, 0] = 5, it means first token is at the 5th entry, if dispatched_indices1 is flatten to 1-D.
    dispatch_indices1 = torch.where(mask_sum1==0, -1, indices1_s*capacity + locations1_s)
    #dim [E * C], each entry stores the token id, the tokens that are not mapped to any expert are stored as the last item
    dispatch_mask[dispatch_indices1] = indices

    mask_sum2 = torch.sum(mask2, dim=1)
    dispatch_indices2 = torch.where(mask_sum2==0, -1, indices2_s*capacity + locations2_s)
    dispatch_mask[dispatch_indices2] = indices + num_tokens #indice + num_tokens* kth top, to make sure each element in the dispatch mask is unique
    dispatch_mask = dispatch_mask[0:-1].reshape(num_experts, -1) #discard the fake tokens

    if fp16_mode is True:
        gates12_s.to(torch.float16)
    return loss, gate_log, gates12_s, dispatch_mask, capacity_fp

class TopKGate(torch.nn.Module):
    """
    The class implements Top1 and Top2 gating function
    Args:
        model_dim (int):
            size of model embedding dimension
        num_experts (int):
            number of experts in model
        balance_ratio (dict[str: float]):
            scaling ratio for the losses (load_balance, sparsity_l1, mean_importance, z_loss, ideal_load_balance. see loss_functions.py for details)
        gate_log_req (dict[str: bool]):
            required gate logs (gate_entropy, gate_probability, gate_routed, expert_fraction, expert_routed_fraction. see gate_logs.py for details)
        capacity_factor (float): 
            a scalar variable to control the cacacity of each expert in training: capacity = num_tokens / number_of_experts * capacity factor
        eval_capacity_factor (float):
            a scalar variable to control the cacacity of each expert in evaluation: capacity = num_tokens / number_of_experts * capacity factor
        k (int): 
            TopK gating function. Currently only supports k = 1 or k = 2
        fp16_mode (bool):
            a boolean variable to control whether fp16_mode is used in moe layer (e.g., by turning on AMP), 
            if so, we cast the inputs and weights in gating function to fp32 for model quality requirement
        switch_jitter (float):
            a small variable to controls the multiplicative jitter to the gate input: x *= uniform(1-epsilon, 1+epsilon)
            only applicable for top1gating
        switch_dropout (float):
            dropout rate for the gate input, only applicable for top1 gating and when switch_jitter is 0
        logits_gumbel (float):
            weight of the Gumbel noise added to the gate logits: logits_w_noise = logits + weight*Gumbel(0, 1)
        random_token_drop (bool):
            whether to randomly drop tokens that are outside the capacity buffer. will be deprecated and replaced with token_drop_type.
        token_drop_type (string):
            how to drop tokens when capacity buffer is not enough: cut, random, routing_weight
        second_place_loss_ratio (float):
            weight of the second place loss for top2gating
        straight_through (bool):
            whether to use Straight Through method to make the load_balance loss fully differentiable
        straight_through_temperature (float):
            temperature of softmax for straight_through
        use_tutel_cumsum_sub_one (callable):
            whether to use fast_cumsum_sub_one from tutel or not
    """
    def __init__(self,
                model_dim: int,
                num_experts: int,
                dgrid,
                balance_ratio=0.01,
                gate_log_req: dict={},
                capacity_factor: float=1.0,
                eval_capacity_factor: float=1.0,
                k: int=2,
                fp16_mode: bool=False,
                switch_jitter: float=0.0,
                switch_dropout: float=0.0,
                logits_gumbel: float=0.0,
                random_token_drop: bool=False,
                token_drop_type: str=None,
                second_place_loss_ratio: float=0.0,
                straight_through: bool=False,
                straight_through_temperature: float=1.0,
                use_tutel_cumsum_sub_one: bool=True,
            ) -> None:
        super().__init__()
        self.is_expert_slicing = dgrid.get_expert_slicing_group() is not None
        self.dgrid = dgrid
        self.wg = torch.nn.Linear(model_dim, num_experts, bias=False)
        self.balance_ratio = balance_ratio_to_dict(balance_ratio)
        self.gate_log_req = gate_log_req
        self.capacity_factor = capacity_factor
        self.eval_capacity_factor = eval_capacity_factor
        self.k = k
        self.fp16_mode = fp16_mode
        self.switch_jitter = switch_jitter
        self.switch_dropout = switch_dropout
        self.logits_gumbel = logits_gumbel
        if token_drop_type is None:
            if random_token_drop:
                self.token_drop_type = 'random'
            else:
                self.token_drop_type = 'cut'
        else:
            assert token_drop_type in ['cut', 'random', 'routing_weight'], f"Unknown token_drop_type '{token_drop_type}', choose among 'cut', 'random', 'routing_weight'."
            self.token_drop_type = token_drop_type
        self.second_place_loss_ratio = second_place_loss_ratio
        self.straight_through = straight_through
        self.straight_through_temperature = straight_through_temperature
        self.loss = None
        self.gate_log = None
        if use_tutel_cumsum_sub_one:
            # from https://github.com/microsoft/tutel (commit e51df1ca64be59eae3691bc1c64b20a201de1009)
            # Please 'run pip install -r ./ requirements.txt' to install tutel
            from tutel.jit_kernels.gating import fast_cumsum_sub_one
            self.tutel_cumsum_sub_one = fast_cumsum_sub_one
        else:
            self.tutel_cumsum_sub_one = None

    def forward(self, input: torch.Tensor, nonpadding: torch.Tensor = None, lid: int=-1) -> Tuple[Tensor, Tensor, float]:  # type: ignore
        """
        Args:
            input (torch.Tensor):
                the input tensor of gating function
            Nonpadding masks (torch.Tensor):
                the nonpadding mask of the input tensor with original shape [s, b].
                In top1gating, only nonpadding tokens are dispatched to experts.
            lid (int):
                the layer id for warning message. Default value -1 will not trigger the warning.
        """
        
        assert self.k ==1 or self.k == 2, "k can only be 1 or 2"
        if self.fp16_mode is True:
            input = input.to(torch.float32)
            self.wg = self.wg.to(torch.float32)
        if self.training and self.k == 1:
            if self.switch_jitter > 0:
                input = multiplicative_jitter(input, device=input.device, epsilon=self.switch_jitter)
            elif self.switch_dropout > 0:
                input = F.dropout(input, p=self.switch_dropout, training=self.training)
        logits = self.wg(input) #dim: [bxs, num_experts]
        if self.k == 1:
            self.loss, self.gate_log, gates1_s, dispatch_mask, retval = top1gating(
                    logits,
                    self.capacity_factor if self.training else self.eval_capacity_factor,
                    is_expert_slicing=self.is_expert_slicing,
                    fp16_mode=self.fp16_mode,
                    nonpadding=nonpadding,
                    logits_gumbel=self.logits_gumbel if self.training else 0,
                    token_drop_type=self.token_drop_type,
                    straight_through=self.straight_through,
                    straight_through_temperature=self.straight_through_temperature,
                    balance_ratio=self.balance_ratio,
                    gate_log_req=self.gate_log_req,
                    lid=lid,
                    tutel_cumsum_sub_one=self.tutel_cumsum_sub_one,
                )
            return gates1_s, dispatch_mask, retval
        else:
            self.loss, self.gate_log, gates12_s, dispatch_mask, capacity_fp = top2gating(
                    logits,
                    self.capacity_factor if self.training else self.eval_capacity_factor,
                    fp16_mode=self.fp16_mode,
                    nonpadding=nonpadding,
                    logits_gumbel=self.logits_gumbel if self.training else 0,
                    token_drop_type=self.token_drop_type,
                    second_place_loss_ratio=self.second_place_loss_ratio,
                    straight_through=self.straight_through,
                    straight_through_temperature=self.straight_through_temperature,
                    balance_ratio=self.balance_ratio,
                    gate_log_req=self.gate_log_req,
                    lid=lid,
                    tutel_cumsum_sub_one=self.tutel_cumsum_sub_one,
                )
            return gates12_s, dispatch_mask, capacity_fp

    def set_gate_metrics(self, balance_ratio=None, gate_log_req=None):
        if balance_ratio is not None:
            self.balance_ratio = balance_ratio
        if gate_log_req is not None:
            self.gate_log_req = gate_log_req

def fast_one_hot(indices: torch.Tensor, num_classes : int):
    assert len(indices.shape) == 1, "indices should only be one dimension"
    device = 'cpu' if indices.get_device() < 0 else indices.get_device()
    ret = torch.zeros(indices.shape[0], num_classes, dtype=torch.int64, device=device)
    ret = ret.scatter(-1, indices.unsqueeze(-1), 1)
    return ret

def top1gating(logits: torch.Tensor, capacity_factor: float, is_expert_slicing=False, fp16_mode: bool=False, nonpadding: torch.Tensor=None,
                logits_gumbel: float=0.0, token_drop_type: str='cut', straight_through: bool=False, straight_through_temperature: float=1.0,
                balance_ratio={'load_balance': 0.01}, gate_log_req: dict={}, lid: int=-1,
                tutel_cumsum_sub_one: callable=None)-> Tuple[Tensor, Tensor, Tensor, Tensor, float]:
    if fp16_mode is True:
        logits = logits.to(torch.float32)

    if logits_gumbel > 0:
        logits_w_noise = logits + logits_gumbel * gumbel_rsample(logits.shape, device=logits.device)

    gates = F.softmax(logits, dim=1) #dim: [bs, num_experts]
    if straight_through:
        if straight_through_temperature != 1.0 or logits_gumbel > 0:
            gates_st = F.softmax((logits_w_noise if logits_gumbel > 0 else logits) / straight_through_temperature, dim=1)
            gates_st_equals_gates = False
        else:
            gates_st = gates
            gates_st_equals_gates = True

    num_tokens = gates.shape[0]
    num_experts = gates.shape[1]
    num_nonpadding = torch.sum(nonpadding) if nonpadding is not None else num_tokens

    if not is_expert_slicing:
        capacity_fp = max(min(num_tokens, (num_tokens / num_experts) * capacity_factor), 4)
        capacity = math.ceil(capacity_fp)

    indices = torch.arange(0, num_tokens, dtype=torch.int32, device=logits.device)

    #create mask for 1st's expert per token
    indices_s = torch.argmax(logits_w_noise if logits_gumbel > 0 else gates, dim = 1) #dim: [bs], the index of the expert with highest softmax value
    mask1 = fast_one_hot(indices_s, num_classes = num_experts) #dim: [bs, num_experts]. 1 for the expert with highest softmax value
    if lid >= 0 and (torch.sum(mask1.float(), dim=0).int() == 0).any():
        print(f"WARNING: top1gating: expert got too few examples in layer {lid}: {torch.sum(mask1.float(), dim=0).int().tolist()}")

    # mask using nonpadding (https://github.com/tensorflow/mesh/blob/a54f5cf75ef44d8a97190b3e5aaec176c138b3c0/mesh_tensorflow/transformer/moe.py#L1224)
    gates_nonpadding = gates
    if nonpadding is not None:
        mask1 = einsum("s,se->se", nonpadding, mask1)
        gates_nonpadding = einsum("s,se->se", nonpadding, gates_nonpadding)
        if straight_through:
            gates_st = einsum("s,se->se", nonpadding, gates_st) if not gates_st_equals_gates else gates_nonpadding
        #TODO: Need to add a unit test
        if is_expert_slicing:
            indices_s = torch.where(nonpadding > 0, indices_s, num_experts) # Assign token_i to "expert num_experts" (fake expert) when nonpadding[i] == 0

    # the fraction of the router probability allocated for each expert
    me = torch.sum(gates_nonpadding, dim=0) / num_nonpadding
    # the fraction of tokens dispatched to each expert
    ce = torch.sum((mask1.float()-gates_st).detach()+gates_st if straight_through else mask1.float(), dim=0) / num_nonpadding

    raw_mask1 = mask1.clone().detach()

    if not is_expert_slicing and token_drop_type in ['random', 'routing_weight']:
        if token_drop_type == 'random':
            # randomly select masked tokens to fit in capacity buffer
            priority = mask1 * torch.rand_like(mask1.float())  # dim: [bs, num_experts]
        else:
            priority = mask1 * gates_nonpadding  # dim: [bs, num_experts]
        _, cap_idx = torch.topk(priority, k=min(capacity, priority.shape[0]), dim=0)  # dim: [capacity, num_experts]
        mask1 = mask1 * torch.zeros_like(mask1).scatter_(0, cap_idx, 1)
        mask1_sum_dim1 = torch.sum(mask1, dim=1)

    if is_expert_slicing:
        expert_count = torch.sum(mask1, dim=0, dtype=torch.int32)

        # indices_s is sorted so that accumulation count for each expert could be obtained without calculating the cumsum of mask1 (cumsum of mask1 is slow)
        # Stable sort is used to make sure that we keep former tokens and drop latter tokens
        shift_count = math.floor(math.log2(max(num_tokens - 1, 1)) + 1)
        experts_sort_info = torch.sort(torch.bitwise_or((indices_s << shift_count), indices))
        experts_sort = experts_sort_info.values >> shift_count
        experts_sort_indices = experts_sort_info.indices

        #TODO: 1. Need more tests for this functionality. Leave it as-is now to unblock CLIP training
        #      2. Need to add a unit test
        if ((not is_expert_slicing) and token_drop_type != 'cut') or nonpadding is not None: # Could be remove in the next version
            discard_tmp = num_tokens - expert_count.sum()
            count_discard = torch.tensor([discard_tmp], device=logits.device)
            expert_count = torch.cat((expert_count, count_discard))

        #stores the token indices for each expert. E.g., if expert_cumcount = [0, 0, 0, 4, 6], that means there are 4 experts, the 1st and 2nd
        #is take in token from input[0 : 0), the second is from input[0:4), the third is from input[4:6)
        expert_cumcount = torch.cumsum(expert_count, dim=0, dtype=torch.int64)
        expert_cumcount = torch.nn.functional.pad(expert_cumcount, (1,0), "constant", 0)

        if is_expert_slicing:
             expert_cumcount_cpu = expert_cumcount.to('cpu')

        # Calculate the accumulation count (indices_in_expert) for each token
        indices_repeat = indices[:, None].expand(num_tokens, num_experts)
        indices_repeat = indices_repeat - expert_cumcount[:num_experts]
        indices_repeat.remainder_(indices[-1] + 1)
        indices_in_expert = torch.min(indices_repeat, dim=1).values
    else:
        #Compute locations in capacity buffer
        if mask1.device.type == 'cpu' or tutel_cumsum_sub_one is None:
            locations1 = torch.cumsum(mask1, dim=0) - 1
        else:
            locations1 = tutel_cumsum_sub_one(mask1, dim=0)

    if not is_expert_slicing and token_drop_type == 'cut':
        #Remove locations outside capacity from mask
        mask1 = mask1 * torch.lt(locations1, capacity)

    # Store the capacity location for each token
    if not is_expert_slicing:
        locations1_s = torch.sum(locations1 * mask1, dim = 1)
        dispatch_mask = torch.full((num_experts * capacity+1,), -1, dtype = int, device=logits.device)
        indices = torch.arange(0, num_tokens, dtype = int, device=logits.device)
        mask_sum = torch.sum(mask1, dim = 1)
        dispatch_indices = torch.where(mask_sum == 0, -1, indices_s*capacity + locations1_s) #dim [E, C], each entry stores the token id, the fake tokens are stored as the last item
        dispatch_mask[dispatch_indices] = indices
        dispatch_mask = dispatch_mask[0:-1].reshape(num_experts, -1) #discard the fake tokens
    else:
        dispatch_mask = torch.full((num_tokens * 1 + 1,), 0, dtype=int, device=logits.device) # in general should be num_tokens * topK
        dispatch_indices = torch.where(experts_sort >= num_experts, -1, expert_cumcount.index_select(0, experts_sort) + indices_in_expert) #dim [S], the index in the dispatch mask for each token.
        dispatch_mask[dispatch_indices] = experts_sort_indices #The token idx sorted following the expert order
        dispatch_mask = dispatch_mask[0:-1] #discard the fake tokens

    #Normalize gate probabilities/ep
    mask1_float = mask1.float()
    gates_nonpadding = gates_nonpadding.float()
    gates1_s = einsum("se,se->s", gates_nonpadding, mask1_float).reshape(1, -1) #[topK, S]

    loss, gate_log = compute_gate_loss(balance_ratio, gate_log_req,
                                        logits=logits, gates=gates_nonpadding, gates_max=gates1_s,
                                        raw_mask=raw_mask1, routing_mask=mask1,
                                        router_prob_fraction=me, token_dispatch_fraction=ce, nonpadding=nonpadding,
                                        num_experts=num_experts, num_nonpadding=num_nonpadding)

    if fp16_mode is True:
        gates1_s = gates1_s.to(torch.float16)
    if not is_expert_slicing:
        # Loss: loss_aux
        # gate_log: the logs of the gating function
        # Gates1_s: dim is [S]. The scaled weight for each token that is allocated to an expert. The same as before.
        # Dispatch_mask: #dim [S].  The expert index for each token in their original sequence order (In expert parallelism, it was cS, where c is a scalar capacity factor.
        # capactiy_fp: The floating point number of capacity
        return loss, gate_log, gates1_s, dispatch_mask, capacity_fp
    else:
        # Loss: loss_aux
        # gate_log: the logs of the gating function
        # Gates1_s: dim is [S]. The scaled weight for each token that is allocated to an expert. The same as before.
        # Dispatch_mask: #dim [S].  The expert index for each token in their original sequence order (In expert parallelism, it was cS, where c is a scalar capacity factor.
        # Expert_cumsum: #dim [number on experts + 1]. This is the prefix of the tokens that mapped to each expert. This is new to the Expert Slicing.
        # Example: Assume the input sequence has length of 4 (S=4): [s0, s1, s2, s3].
        # Assume there are 3 experts (e0, e1, e2). s0 maps to expert e1, s1 maps to expert e2, s2 maps to e2 and s3 maps to e1. So, the dispatch mask is [1, 2, 2, 1].
        # The expert_cumsum is [0, 0, 2, 4]. (e0 is from [0:0], e1 is from [0:2], e2 is from [2:4])
        return loss, gate_log, gates1_s, dispatch_mask, expert_cumcount_cpu

def balance_ratio_to_dict(balance_ratio):
    # convert float and list balance_ratio to dictionary to be backward compatible
    if isinstance(balance_ratio, dict):
        for k in balance_ratio.keys():
            assert k in loss_functions, f"Unkonwn balance ratio '{k}'"
    elif isinstance(balance_ratio, float) or isinstance(balance_ratio, int):
        balance_ratio = {'load_balance': balance_ratio}
    elif isinstance(balance_ratio, list):
        # assume the orders are the same
        num_losses = min(len(loss_functions), len(balance_ratio))
        balance_ratio_dict = {}
        for k, v in zip(list(loss_functions.keys())[:num_losses], balance_ratio[:num_losses]):
            balance_ratio_dict[k] = v
        balance_ratio = balance_ratio_dict
    else:
        raise f"Unknown balance_ratio format: {balance_ratio}"

    print(f"MoE loss balance ratio: {balance_ratio}")
    return balance_ratio

def compute_gate_loss(balance_ratio, gate_log_req, **kwargs):
    # compute gate loss and other logging information
    loss = torch.empty([], dtype=kwargs['logits'].dtype, device=kwargs['logits'].device).fill_(0)
    gate_log = {}

    for type, ratio in balance_ratio.items():
        if ratio > 0:
            l = loss_functions[type](**kwargs) * ratio
            gate_log[type] = l
            loss += l

    for type, required in gate_log_req.items():
        if required:
            gate_log[type] = gate_logs[type](**kwargs)

    return loss, gate_log
