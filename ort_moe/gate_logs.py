# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import torch

def gate_entropy(gates=None, num_nonpadding=None, **kwargs):
    # the entropy of the router distribution
    # https://github.com/tensorflow/mesh/blob/a54f5cf75ef44d8a97190b3e5aaec176c138b3c0/mesh_tensorflow/transformer/moe.py#L1241
    # gates: [bs, num_experts]
    assert gates is not None and num_nonpadding is not None, f"gate_entropy needs 'gates' and 'num_nonpadding' as input argument."
    expert_entropy = torch.sum(-gates * torch.log(gates + 1e-9), dim=1)
    return torch.sum(expert_entropy) / num_nonpadding

def gate_probability(gates_max=None, num_nonpadding=None, **kwargs):
    # the probability of the selected expert, only nonpadding tokens being considered
    # https://github.com/tensorflow/mesh/blob/a54f5cf75ef44d8a97190b3e5aaec176c138b3c0/mesh_tensorflow/transformer/moe.py#L1243
    # gates_max: [bs]
    assert gates_max is not None and num_nonpadding is not None, f"gate_probability needs 'gates_max' and 'num_nonpadding' as input argument."
    return torch.sum(gates_max) / num_nonpadding

def gate_routed(routing_mask=None, num_nonpadding=None, **kwargs):
    # routed token fraction of the entire gate
    # https://github.com/tensorflow/mesh/blob/a54f5cf75ef44d8a97190b3e5aaec176c138b3c0/mesh_tensorflow/transformer/moe.py#L1284
    # routing_mask: [bs, num_experts]
    assert routing_mask is not None and num_nonpadding is not None, f"gate_routed needs 'routing_mask' and 'num_nonpadding' as input arguments."
    routed_count_experts = torch.sum(routing_mask, dim=0)
    return torch.sum(routed_count_experts) / num_nonpadding

def expert_fraction(raw_mask=None, **kwargs):
    # the fraction of tokens dispatched to each expert
    # https://github.com/tensorflow/mesh/blob/a54f5cf75ef44d8a97190b3e5aaec176c138b3c0/mesh_tensorflow/transformer/moe.py#L1247
    # raw_mask: [bs, num_experts]
    assert raw_mask is not None, f"expert_fraction needs 'raw_mask' as input argument."
    mask_count_experts = torch.sum(raw_mask, dim=0)
    total_masks = torch.sum(mask_count_experts)
    return mask_count_experts.float() / total_masks  # dim: [num_experts]

def expert_routed_fraction(routing_mask=None, **kwargs):
    # routed token fraction of each expert
    # https://github.com/tensorflow/mesh/blob/a54f5cf75ef44d8a97190b3e5aaec176c138b3c0/mesh_tensorflow/transformer/moe.py#L1284
    # routing_mask: [bs, num_experts]
    assert routing_mask is not None, f"expert_routed_fraction needs 'gates_max' as input argument."
    routed_count_experts = torch.sum(routing_mask, dim=0)
    total_routed = torch.sum(routed_count_experts)
    return routed_count_experts.float() / total_routed  # dim: [num_experts]

gate_logs = {
    'gate_entropy': gate_entropy,
    'gate_probability': gate_probability,
    'gate_routed': gate_routed,
    'expert_fraction': expert_fraction,
    'expert_routed_fraction': expert_routed_fraction,
}
