# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import TYPE_CHECKING, Any, Optional, Tuple
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import math
import os
from ort_moe import experts
from ort_moe import utils

class MixtureOfExperts(nn.Module):
    r"""MixtureOfExperts module implements mixture of experts.
    Args:
        gate: the gating function (required).
        experts: list of experts (required).
        ep_group: The process group to be used to distribute experts. If ``None``, the
                       the default process group provided by torch.distributed will be used.
                       (default=None)
        balance_ratio: The scaling ratio for the loss_aux
        is_encoder: Whether this MOE is in encoder layer. If false, it is decoder layer
    """
    #max_len dictionary for encoder and decoder.
    #Need two dicts because the max_len for encoder and decoder may be different
    #In the forward method, based on the self.is_encoder, it feeds the corresponding max_len dict to AllToAll
    #TODO: This is not a good solution, since this becomes a static member of MoE class, if different encoder/decoder
    #layers have different length, this will break
    enc_max_len ={"max_len": None, "capacity_fp": 0, "need_update": True}
    dec_max_len ={"max_len": None, "capacity_fp": 0, "need_update": True}

    #Destructor to reset the max_len dictionary
    def __del__(self):
        self.reset_moe_state()

    @classmethod
    def reset_moe_encoder_state(cls):
        cls.enc_max_len["need_update"] = True

    @classmethod
    def reset_moe_decoder_state(cls):
        cls.dec_max_len["need_update"] = True

    @classmethod
    def reset_moe_state(cls):
        cls.reset_moe_encoder_state()
        cls.reset_moe_decoder_state()

    def __init__(self, gate, experts_, ep_group=None, balance_ratio = [0.01], is_encoder = True, fp16_mode = False, use_mpi4py = True):
        super(MixtureOfExperts, self).__init__()
        self.is_mergedFFNExpert = isinstance(experts_, experts.MergedFFNExpert)

        self.gate = gate
        self.experts = experts_

        self.num_experts = self.experts.local_num_experts if self.is_mergedFFNExpert else len(experts_)
        self.balance_ratio = balance_ratio
        self.l_aux = []
        self.is_encoder = is_encoder
        self.use_mpi4py = use_mpi4py

        if(dist.is_initialized()):
            if self.use_mpi4py:
                from mpi4py import MPI
                if ep_group is None:
                    self.mpi_ep_group = MPI.COMM_WORLD
                else:
                    for g in ep_group:
                        tmp = MPI.COMM_WORLD.Create_group(MPI.COMM_WORLD.group.Incl(g))
                        if MPI.COMM_WORLD.Get_rank() in g:
                            self.mpi_ep_group = tmp

            if ep_group is None:
                self.ep_group = dist.group.WORLD
            else: # need to create all groups, even the one it doesn't belong to
                for g in ep_group:
                    tmp = dist.new_group(g)
                    if dist.get_rank(dist.group.WORLD) in g:
                        self.ep_group = tmp

            self.ep_group_size = dist.get_world_size(self.ep_group)
        else:
            self.ep_group = None
            self.ep_group_size = 1

        #tag the is_moe_param for the experts, later in the application people can extract expert specific parameters if needed
        for p in self.experts.parameters():
            p.is_moe_param = True

        #set the parameter list for experts and non experts
        self._expert_parameters_list = []
        self._non_expert_parameters_list = []
        for p in self.parameters():
            if utils.is_moe_parameter(p):
                self._expert_parameters_list.append(p)
            else:
                self._non_expert_parameters_list.append(p)
        self.fp16_mode = fp16_mode

    def get_max_len(self, input: Tensor, max_len, capacity_fp):
        if not math.isclose(capacity_fp,max_len["capacity_fp"]):
            max_len["capacity_fp"] = capacity_fp
            max_len["need_update"] = True
        if max_len["need_update"]:
            if self.ep_group_size == 1:
                max_len["max_len"] = input.shape[-2]
                max_len["need_update"] = False
            else:
                max_len_tensor = input.shape[-2]
                if self.use_mpi4py:
                    from mpi4py import MPI
                    max_len_tensor = self.mpi_ep_group.allreduce(max_len_tensor, MPI.MAX)
                    max_len["max_len"] = max_len_tensor
                else:
                    local_rank = input.get_device()
                    max_len_tensor = torch.tensor(input.shape[-2]).to(local_rank)
                    dist.all_reduce(max_len_tensor, op=dist.ReduceOp.MAX, group=self.ep_group)
                    max_len["max_len"] = max_len_tensor.item()

                max_len["need_update"] = False

    def forward(self, *input:Tensor, **kwargs:Any) -> Tensor:
        assert len(input[0].shape) == 3, "input Tensor must have dimensions: (s)equence, (t)oken, (m)odel"
        assert len(input) == 1, "only single input Tensor supported"
        max_len = self.enc_max_len if self.is_encoder else self.dec_max_len

        # Implement Algorithm 2 from GShard paper.
        d_model = input[0].shape[2]
        # Reshape into S tokens by dropping sequence dimension.
        reshaped_input = input[0].reshape(-1, d_model)
        # Nonpadding masks of the input tensor with original shape [s, b].
        # In top1gating, only nonpadding tokens are dispatched to experts.
        reshaped_nonpadding = kwargs['nonpadding'].reshape(-1) if kwargs.get('nonpadding', None) is not None else None
        self.l_aux, combine_weights, dispatch_mask, capacity_fp = self.gate(reshaped_input, nonpadding=reshaped_nonpadding)
        # l_aux is a list of metrics: [l_aux, l_sl1, l_mil, l_z, batch_entropy, batch_prob, fraction_routed, [expert_fraction], [expert_fraction_routed]]
        # l_* metrics are balanced with self.balance_ratio and contribute to the final loss.
        # Other metrics are for logging and visualization only.
        for i in range(min(self.l_aux.shape[0], len(self.balance_ratio))):
            self.l_aux[i] *= self.balance_ratio[i]
        dispatched_input = torch.einsum("sec,sm->ecm", dispatch_mask.to(reshaped_input), reshaped_input)
        if self.fp16_mode and dispatched_input.dtype is not torch.float16:
            dispatched_input = dispatched_input.to(torch.float16)
        c_cpu = dispatched_input.shape[1]
        if self.ep_group_size > 1:
            #add the allreduce to get the max_len
            self.get_max_len(dispatched_input, max_len, capacity_fp)
            assert max_len["max_len"] >= c_cpu
            dispatched_input = _AllToAll.apply(self.ep_group, dispatched_input, max_len['max_len'])
        # Re-shape after all-to-all: ecm -> gecm
        dispatched_input = dispatched_input.reshape(self.ep_group_size, self.num_experts, -1, d_model)

        if not self.is_mergedFFNExpert:
            chunks = dispatched_input.chunk(self.num_experts, dim=1)
            expert_outputs = []
            for chunk, expert in zip(chunks, self.experts):
                expert_outputs += [expert(chunk)]
            expert_output = torch.cat(expert_outputs, dim=1)
        else:
            expert_output = self.experts(dispatched_input)

        if self.ep_group_size > 1:
            expert_output = _AllToAll.apply(self.ep_group, expert_output, max_len['max_len'])
        expert_output = torch.narrow(expert_output, dim = 2, start=0, length = c_cpu)
        # Re-shape back: gecm -> ecm
        expert_output = expert_output.reshape(self.ep_group_size * self.num_experts, -1, d_model)
        combined_output = torch.einsum("sec,ecm->sm", combine_weights, expert_output)
        return combined_output.reshape(input[0].shape)

    #return the list of expert parameters
    @property
    def expert_parameters_list(self):
        return self._expert_parameters_list

    #return the state dict of expert parameters
    @property
    def expert_parameters_state_dict_keys(self):
        return self._expert_parameters_sdict_keys

    #return the list of non-expert parameters
    @property
    def non_expert_parameters_list(self):
        return self._non_expert_parameters_list

    #return the state dict of non-expert parameters
    @property
    def non_expert_parameters_state_dict_keys(self):
        return self._non_expert_parameters_sdict_keys

# Based on https://github.com/pytorch/pytorch/pull/40762
class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, input: Tensor, max_len_cpu) -> Tensor:  # type: ignore
        ctx.group = group
        ctx.input_c = input.shape[-2]
        ctx.max_len_cpu = max_len_cpu

        #pad the input to the max_len
        assert max_len_cpu >= input.shape[-2]
        if max_len_cpu != input.shape[-2]:
            input = torch.nn.functional.pad(input, pad=(0,0,0,max_len_cpu-input.shape[-2]), mode = 'constant', value=0.0)
        input = input.contiguous()
        output = torch.empty_like(input)
        dist.all_to_all_single(output, input, group=group)
        return output

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor]:
        input_c = ctx.input_c
        assert ctx.max_len_cpu >= input_c
        ret_val = _AllToAll.apply(ctx.group, *grad_output, ctx.max_len_cpu)
        narrow_dim = len(ret_val.shape)-2
        ret_val = torch.narrow(ret_val, dim = narrow_dim, start=0, length = input_c)
        return (None, ret_val, None)