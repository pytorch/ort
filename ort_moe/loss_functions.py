# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from collections import OrderedDict

import torch

def load_balancing_loss(router_prob_fraction=None, token_dispatch_fraction=None, num_experts=None, **kwargs):
    """
    the load balancing loss defined in Switch Transformers: https://arxiv.org/abs/2101.03961
    Args:
        router_prob_fraction: [num_experts]
        token_dispatch_fraction: [num_experts]
        num_experts: The number of experts
    """
    assert router_prob_fraction is not None and token_dispatch_fraction is not None and num_experts is not None, \
        f"load_ballancing_loss needs 'router_prob_fraction', 'token_dispatch_fraction' and 'num_experts' as input arguments."
    return torch.sum(router_prob_fraction * token_dispatch_fraction) * num_experts

def sparsity_l1_loss(gates=None, num_nonpadding=None, **kwargs):
    """
    the sparsity L1 loss defined in SpeechMoE: https://arxiv.org/abs/2105.03036
    Args:
        gates: [bs, num_experts]
        num_nonpadding: The number of nonpadding tokens
    """
    assert gates is not None and num_nonpadding is not None, f"sparsity_l1_loss needs 'gates' and 'num_nonpadding' as input argument."
    return torch.sum(torch.norm(gates / (torch.norm(gates, p=2, dim=1, keepdim=True) + 1e-9), p=1, dim=1)) / num_nonpadding

def mean_importance_loss(router_prob_fraction=None, num_experts=None, **kwargs):
    """
    the mean importance loss defined in SpeechMoE: https://arxiv.org/abs/2105.03036
    Args:
        router_prob_fraction: [num_experts]
        num_experts: The number of experts
    """
    assert router_prob_fraction is not None and num_experts is not None, \
        f"mean_importance_loss needs 'router_prob_fraction' and 'num_experts' as input arguments."
    return torch.sum(torch.mul(router_prob_fraction, router_prob_fraction)) * num_experts

def z_loss(logits=None, nonpadding=None, num_nonpadding=None, **kwargs):
    """
    z_loss: https://github.com/tensorflow/mesh/blob/a54f5cf75ef44d8a97190b3e5aaec176c138b3c0/mesh_tensorflow/transformer/moe.py#L1258
    Args:
     logits: [bs, num_experts]
     nonpadding: [bs]
     num_nonpadding: The number of non_padding tokens
    """
    assert logits is not None and num_nonpadding is not None, \
        f"z_loss needs 'logits' and 'num_nonpadding' as input arguments."
    loss = torch.square(torch.logsumexp(logits, dim=1))
    if nonpadding is not None:
        loss *= nonpadding
    return torch.sum(loss) / num_nonpadding

def ideal_load_balancing_loss(token_dispatch_fraction=None, num_experts=None, **kwargs):
    """
    The ideal load balancing loss which is originally not differentiable.
    To make it differentiable, set STRAIGHT_THROUGH = True.
    There is another parameter to control the softmax temperature.
    The smaller the STRAIGHT_THROUGH_TEMPERATURE, the closer the distribution to one-hot.
    TODO: decrease the temperature using scheduler.
    Note that Switch Transformers (https://arxiv.org/abs/2101.03961) uses a differentiable approximation of it.
    Args:
        token_dispatch_fraction: [num_experts]
        num_experts: The number of experts
    """
    assert token_dispatch_fraction is not None and num_experts is not None, \
        f"ideal_load_balancing_loss needs 'token_dispatch_fraction' and 'num_experts' as input arguments."
    return torch.sum(token_dispatch_fraction * token_dispatch_fraction) * num_experts

loss_functions = OrderedDict({
    'load_balance': load_balancing_loss,
    'sparsity_l1': sparsity_l1_loss,
    'mean_importance': mean_importance_loss,
    'z_loss': z_loss,
    'ideal_load_balance': ideal_load_balancing_loss,
})
