# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
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
    minval = torch.tensor(1.0 - epsilon, device=device)
    maxval = torch.tensor(1.0 + epsilon, device=device)
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

def top2gating(logits: torch.Tensor, fp16_mode: bool = False) -> Tuple[Tensor, Tensor, Tensor, float]:
    """Implements Top2Gating on logits."""
    if fp16_mode is True:
        logits = logits.to(torch.float32)
    gates = F.softmax(logits, dim=1)

    # gates has shape of SE
    num_tokens = gates.shape[0]
    num_experts = gates.shape[1]
    # capacity = 2S/E
    capacity_fp = 2 * num_tokens / num_experts
    capacity = math.ceil(capacity_fp)
    assert num_tokens % num_experts == 0

    # Create a mask for 1st's expert per token
    indices1_s = torch.argmax(gates, dim=1)
    mask1 = F.one_hot(indices1_s, num_classes=num_experts)

    # Create a mask for 2nd's expert per token using Gumbel-max trick
    # https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/
    logits_w_noise = logits + gumbel_rsample(logits.shape, device=logits.device)
    # Replace top-expert with min value
    logits_except1 = logits_w_noise.masked_fill(mask1.bool(), float("-inf"))
    indices2_s = torch.argmax(logits_except1, dim=1)
    mask2 = F.one_hot(indices2_s, num_classes=num_experts)

    # Compute locations in capacity buffer
    locations1 = torch.cumsum(mask1, dim=0) - 1
    locations2 = torch.cumsum(mask2, dim=0) - 1
    # Update 2nd's location by accounting for locations of 1st
    locations2 += torch.sum(mask1, dim=0, keepdim=True)

    # Compute l_aux
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask1.float(), dim=0)
    l_aux = torch.sum(me * ce) * num_experts
    l_aux = torch.stack((l_aux, l_aux*0, l_aux*0, l_aux*0))

    # Remove locations outside capacity from mask
    mask1 *= torch.lt(locations1, capacity)
    mask2 *= torch.lt(locations2, capacity)

    # Store the capacity location for each token
    locations1_s = torch.sum(locations1 * mask1, dim=1)
    locations2_s = torch.sum(locations2 * mask2, dim=1)

    # Normalize gate probabilities
    mask1_float = mask1.float()
    mask2_float = mask2.float()
    gates1_s = torch.einsum("se,se->s", gates, mask1_float)
    gates2_s = torch.einsum("se,se->s", gates, mask2_float)
    denom_s = gates1_s + gates2_s
    # Avoid divide-by-zero
    denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
    gates1_s /= denom_s
    gates2_s /= denom_s

    # Calculate combine_weights and dispatch_mask
    gates1 = torch.einsum("s,se->se", gates1_s, mask1_float)
    gates2 = torch.einsum("s,se->se", gates2_s, mask2_float)
    locations1_sc = F.one_hot(locations1_s, num_classes=capacity)
    locations2_sc = F.one_hot(locations2_s, num_classes=capacity)
    combine1_sec = torch.einsum("se,sc->sec", gates1, locations1_sc)
    combine2_sec = torch.einsum("se,sc->sec", gates2, locations2_sc)
    combine_weights = combine1_sec + combine2_sec
    dispatch_mask = combine_weights.to(torch.bool)
    if fp16_mode is True:
        combine_weights = combine_weights.to(torch.float16)
    return l_aux, combine_weights, dispatch_mask, capacity_fp


class TopKGate(torch.nn.Module):
    """Gate module which implements Top2Gating as described in Gshard_.
    ::

        gate = Top2Gate(model_dim, num_experts)
        l_aux, combine_weights, dispatch_mask = gate(input)

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        model_dim (int):
            size of model embedding dimension
        num_experts (ints):
            number of experts in model
        switch_jitter (float):
            a small variable to controls the multiplicative jitter to the gate input: x *= uniform(1-epsilon, 1+epsilon)
            only applicable for top1gating
        switch_dropout (float):
            dropout rate for the gate input, only applicable for top1 gating and when switch_jitter is 0
        random_token_drop (bool):
            whether to randomly drop tokens that are outside the capacity buffer
    """

    def __init__(self,
                model_dim: int,
                num_experts: int,
                capacity_factor: float = 1.0,
                eval_capacity_factor: float = 1.0,
                k: int = 2,
                fp16_mode: bool = False,
                switch_jitter: float = 0.0,
                switch_dropout: float = 0.0,
                random_token_drop: bool = False,
            ) -> None:
        super().__init__()
        self.wg = torch.nn.Linear(model_dim, num_experts, bias=False)
        self.capacity_factor = capacity_factor
        self.eval_capacity_factor = eval_capacity_factor
        self.k = k
        self.fp16_mode = fp16_mode
        self.switch_jitter = switch_jitter
        self.switch_dropout = switch_dropout
        self.random_token_drop = random_token_drop

    def forward(self, input: torch.Tensor, nonpadding: torch.Tensor = None) -> Tuple[Tensor, Tensor, Tensor, float]:  # type: ignore
        if self.fp16_mode is True:
            input = input.to(torch.float32)
            self.wg = self.wg.to(torch.float32)
        if self.training and self.k == 1:
            if self.switch_jitter > 0:
                input = multiplicative_jitter(input, device=input.device, epsilon=self.switch_jitter)
            elif self.switch_dropout > 0:
                input = F.dropout(input, p=self.switch_dropout, training=self.training)
        logits = self.wg(input) #dim: [bxs, num_experts]
        assert self.k ==1 or self.k == 2, "k can only be 1 or 2"
        if self.k == 1:
            return top1gating(logits, self.capacity_factor if self.training else self.eval_capacity_factor, self.fp16_mode, nonpadding, self.random_token_drop)
        else:
            return top2gating(logits, self.fp16_mode)


def fast_one_hot(indices: torch.Tensor, num_classes : int):
    assert len(indices.shape) == 1, "indices should only be one dimension"
    device = 'cpu' if indices.get_device() < 0 else indices.get_device()
    ret = torch.zeros(indices.shape[0], num_classes, dtype=torch.int64, device=device)
    ret = ret.scatter(-1, indices.unsqueeze(-1), 1)
    return ret

def top1gating(logits: torch.Tensor, capacity_factor: float, fp16_mode: bool = False, nonpadding: torch.Tensor = None, random_token_drop: bool = False)-> Tuple[Tensor, Tensor, Tensor, float]:
    if fp16_mode is True:
        logits = logits.to(torch.float32)
    gates = F.softmax(logits, dim=1) #dim: [bs, num_experts]
    num_tokens = gates.shape[0]
    num_experts = gates.shape[1]
    num_nonpadding = torch.sum(nonpadding) if nonpadding is not None else num_tokens
    capacity_fp = max(min(num_tokens, (num_tokens / num_experts) * capacity_factor), 4)
    capacity = math.ceil(capacity_fp)

    #create mask for 1st's expert per token
    gates_s, indices_s = torch.max(gates, dim = 1) #dim: [bs], the index of the expert with highest softmax value
    mask1 = fast_one_hot(indices_s, num_classes = num_experts) #dim: [bs, num_experts]. 1 for the expert with highest softmax value

    #Compute l_aux
    me = torch.mean(gates, dim=0)
    # mask using nonpadding (https://github.com/tensorflow/mesh/blob/a54f5cf75ef44d8a97190b3e5aaec176c138b3c0/mesh_tensorflow/transformer/moe.py#L1224)
    if nonpadding is not None:
        gates_s *= nonpadding
        mask1 = torch.einsum("s,se->se", nonpadding, mask1)
    ce = torch.mean(mask1.float(), dim=0)
    l_aux = torch.sum(me * ce) * num_experts
    # sparsity L1 loss and mean importance loss defined in SpeechMoE (https://arxiv.org/abs/2105.03036)
    l_sl1 = torch.mean(torch.norm(gates / (torch.norm(gates, p=2, dim=1, keepdim=True) + 1e-9), p=1, dim=1))
    l_mil = torch.sum(torch.mul(me, me)) * num_experts

    # entropy (https://github.com/tensorflow/mesh/blob/a54f5cf75ef44d8a97190b3e5aaec176c138b3c0/mesh_tensorflow/transformer/moe.py#L1241)
    expert_entropy = torch.sum(-gates * torch.log(gates + 1e-9), dim=1)
    batch_entropy = torch.mean(expert_entropy)
    # probability (https://github.com/tensorflow/mesh/blob/a54f5cf75ef44d8a97190b3e5aaec176c138b3c0/mesh_tensorflow/transformer/moe.py#L1243)
    batch_prob = torch.mean(gates_s)

    # expert (https://github.com/tensorflow/mesh/blob/a54f5cf75ef44d8a97190b3e5aaec176c138b3c0/mesh_tensorflow/transformer/moe.py#L1247)
    mask_count_experts = torch.sum(mask1, dim=0)
    total_masks = torch.sum(mask_count_experts)
    expert_fraction = mask_count_experts.float() / total_masks

    # z_loss (https://github.com/tensorflow/mesh/blob/a54f5cf75ef44d8a97190b3e5aaec176c138b3c0/mesh_tensorflow/transformer/moe.py#L1258)
    l_z = torch.square(torch.logsumexp(logits, dim=1))
    if nonpadding is not None:
        l_z *= nonpadding
    l_z = torch.sum(l_z) / num_nonpadding

    if random_token_drop:
        # randomly select masked tokens to fit in capacity buffer
        mask1_rand = mask1 * torch.rand_like(mask1.float())
        _, cap_idx = torch.topk(mask1_rand, k=min(capacity, mask1_rand.shape[0]), dim=0)
        mask1 = mask1 * torch.zeros_like(mask1).scatter_(0, cap_idx, 1)

    #Compute locations in capacity buffer
    locations1 = torch.cumsum(mask1, dim=0)-1

    if not random_token_drop:
        #Remove locations outside capacity from mask
        mask1 *= torch.lt(locations1, capacity)

    # routed fraction (https://github.com/tensorflow/mesh/blob/a54f5cf75ef44d8a97190b3e5aaec176c138b3c0/mesh_tensorflow/transformer/moe.py#L1284)
    routed_count_experts = torch.sum(mask1, dim=0)
    total_routed = torch.sum(routed_count_experts)
    expert_fraction_routed = routed_count_experts.float() / total_routed
    fraction_routed = total_routed / num_nonpadding

    #Store the capacity location for each token
    locations1_s = torch.sum(locations1 * mask1, dim = 1)

    # l_* may contribute to the final loss depending on the balance_ratio list
    l_aux = torch.cat((torch.stack((l_aux, l_sl1, l_mil, l_z, batch_entropy, batch_prob, fraction_routed)), expert_fraction, expert_fraction_routed))

    #Normalize gate probabilities
    mask1_float = mask1.float()
    gates = gates.float()
    gates1_s = torch.einsum("se, se->s", gates, mask1_float) #dim: [num_tokens]

    #Calculate combine_weights and dispatch_mask
    gates1 = torch.einsum("s,se->se", gates1_s, mask1_float) #dim: [num_tokens, experts]
    locations1_sc = fast_one_hot(locations1_s, num_classes=capacity)
    combine_weights = torch.einsum("se,sc->sec", gates1, locations1_sc) #dim[num_tocken, exeprts, capacity]
    dispatch_mask = combine_weights.to(torch.bool)
    if fp16_mode is True:
        combine_weights = combine_weights.to(torch.float16)
    return l_aux, combine_weights, dispatch_mask, capacity_fp
