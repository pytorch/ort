# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import torch
from .moe_config import moe_config

# The switch to decided whether to use torch.einsum (when this flag is true) or use rewrited-version.
# switch can be bubbled up in future
USE_EINSUM = True

def om_einsum(rule, a, b):
    """
    The rewrite of torch.einsum for some specific cases.
        The rewrites are on par or more performant upon the benchmark we tested
    Args:
        rule (string): the set of einsum rules. 
        a (torch.Tensor): the 1st input tensor of einsum op
        b (torch.Tensor): the 2nd input tensor of einsum op
    """
    if USE_EINSUM:
        return torch.einsum(rule, a, b)
    elif rule == 's,se->se':
        return a.reshape(a.shape[0], -1) * b
    elif rule == 'se,sc->sec':
        return a.unsqueeze(2) * b.unsqueeze(1)
    elif rule == 'se,se->s':
        return torch.bmm(a.unsqueeze(1), b.unsqueeze(2)).reshape(-1)
    elif rule == 'sec,sm->ecm':
        s = a.shape[0]
        e = a.shape[1]
        c = a.shape[2]
        m = b.shape[1]
        return torch.matmul(a.reshape(s, -1).t(), b).reshape(e, c, m)
    elif rule == 'ks,ksm->sm':
        k = b.shape[0]
        s = b.shape[1]
        m = b.shape[2]
        # [k, s] -> [s, k] -> [s, 1, k]
        a = a.t().unsqueeze(1)
        # [k,s,m] -> [k, sm] -> [sm, k] -> [s, m, k]
        b = b.reshape(k, -1).t().reshape(s, m, k)
        # bmm([s, 1, k], [s, m, k]^t) -> [s, m, 1]
        return torch.bmm(a, b.transpose(1, 2)).reshape(s, m)
    else:
        return torch.einsum(rule, a, b)

def om_cumsum(mask, dim, options = None):
    """
    The rewrite of torch.cumsum to use tutel cumsum if desired.
    Args:
        tensor (torch.Tensor): the input tensor of cumsum op
        dim (int): the dimension of cumsum op
        options (moe_config): the options to decide whether to use tutel cumsum
    """
    if mask.device.type == 'cpu' or options is None:
        return torch.cumsum(mask, dim) - 1

    moe_options = None
    if isinstance(options, moe_config): moe_options = options
    else:  moe_options = moe_config(options)
    
    if moe_options.enable_tutel_cumsum():
        from tutel.jit_kernels.gating import fast_cumsum_sub_one
        return fast_cumsum_sub_one(mask, dim)

    return torch.cumsum(mask, dim) - 1
