# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
#
# The file has been adapted from two fairscale files:
# (1) https://github.com/facebookresearch/fairscale/blob/master/fairscale/nn/moe/moe_layer.py
# (2) https://github.com/facebookresearch/fairscale/blob/master/fairscale/nn/moe/top2gate.py
# We retain the following license from the original files:

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any
import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.distributed as dist
import torch.utils.checkpoint as checkpoint
from . import experts
from .moe_config import moe_config
from .collectives import AllGather, AllToAll, AllReduce
from .custom_ops import om_einsum

def MixtureOfExpertsFunc(gate, experts_, distribution_grid, is_encoder = True, options = {}):
    r"""A factory function to call different MoE classes
    Args:
        gate: the gating function (required).
        experts_: list of experts (required).
        distribution_grid: DistributionGrid object providing interface to query torch.distributed process groups.
            It is a required keyword to remove the confusion for the usage of dgrid, for single GPU, instantiate an empty dgrid()
        is_encoder: Whether this MOE is in encoder layer. If false, it is decoder layer
        options: See moe_config.py
    """
    if distribution_grid.get_expert_slicing_group() is not None:
        return MixtureOfExpertsES(gate, experts_, distribution_grid, is_encoder, options)
    else:
        return MixtureOfExpertsEP(gate, experts_, distribution_grid, is_encoder, options)

class MixtureOfExperts(nn.Module):
    r"""MixtureOfExperts module implements mixture of experts.
    Args:
        gate: the gating function (required).
        experts_: list of experts (required).
        distribution_grid: DistributionGrid object providing interface to query torch.distributed process groups.
            It is a required keyword to remove the confusion for the usage of dgrid, for single GPU, instantiate an empty dgrid()
        is_encoder: Whether this MOE is in encoder layer. If false, it is decoder layer
        options: See moe_config.py
        """
    #max_len dictionary for encoder and decoder.
    #Need two dicts because the max_len for encoder and decoder may be different
    #In the forward method, based on the self.is_encoder, it feeds the corresponding max_len dict to AllToAll
    #TODO: This is not a good solution, since this becomes a static member of MoE class
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

    def __init__(self, gate, experts_, distribution_grid, is_encoder = True, options = {}):
        super(MixtureOfExperts, self).__init__()
        assert distribution_grid != None
        self._options = moe_config(options)
        self.is_mergedFFNExpert = isinstance(experts_, experts.MergedFFNExpert)
        self.gate = gate
        self.moe_experts = experts_

        self.num_experts = self.moe_experts.local_num_experts if self.is_mergedFFNExpert else len(experts_)
        self.is_encoder = is_encoder

        self.expert_rank = distribution_grid.get_expert_rank()
        self.expert_group = distribution_grid.get_expert_group()
        self.expert_group_size = distribution_grid.get_expert_world_size()
        if self._options.use_mpi_for_imbalanced_input():
            self.mpi_expert_group = distribution_grid.get_mpi_group_for_expert_group()

        #tag the is_moe_param for the experts, later in the application people can extract expert specific parameters if needed
        for p in self.moe_experts.parameters():
            p.is_moe_param = True
            if self._options.enable_deepspeed_zero_optimization():
                # Allreduce and group_name attributes are for DS Zero 1,2,3
                p.allreduce = False
                p.group_name = f"ep_size_{self.expert_group_size}"

        for p in self.gate.parameters():
            p.is_gate_param = True

    def get_max_len(self, tensor_len, max_len, device):
        r"""To obtain the maximum length of the tensor's specific dimension, store it in max_len dictionary. This is later used in alltoall
        to pad all tensors to the maximum length.
        Args:
            tensor_len: for a given tensor, the length of the dimension that is to be padded
            max_len: the dictionary to store the max lengths of a set of tensor.
            device: the device the input tensor resides on
        """
        if max_len["need_update"]:
            if self.expert_group_size == 1:
                max_len["max_len"] = tensor_len
                max_len["need_update"] = False
            else:
                max_len_tensor = tensor_len
                if self._options.use_mpi_for_imbalanced_input():
                    from mpi4py import MPI
                    max_len_tensor = self.mpi_expert_group.allreduce(max_len_tensor, MPI.MAX)
                    max_len["max_len"] = max_len_tensor
                else:
                    max_len_tensor = torch.empty([], device = device, dtype=type(max_len_tensor)).fill_(max_len_tensor)
                    dist.all_reduce(max_len_tensor, op=dist.ReduceOp.MAX, group=self.expert_group)
                    max_len["max_len"] = max_len_tensor.item()

                max_len["need_update"] = False

    def forward(self, input):
        raise NotImplementedError("Should not directly instantiate the MixtureOfExpert class, this is a base class for MixtureOfExpertsES and \
        MixtureOfExpertsEP, instead, call the factory function MixtureOfExpertFunc")

class MixtureOfExpertsES(MixtureOfExperts):
    def __init__(self, gate, experts_, distribution_grid, is_encoder = True, options = {}):
        r"""MixtureOfExpertsES module implements mixture of experts with expert slicing
        Args:
            gate: the gating function (required).
            experts_: list of experts (required).
            distribution_grid: DistributionGrid object providing interface to query torch.distributed process groups.
                It is a required keyword to remove the confusion for the usage of dgrid, for single GPU, instantiate an empty dgrid()
            is_encoder: Whether this MOE is in encoder layer. If false, it is decoder layer
            options: See moe_config.py
        """
        MixtureOfExperts.__init__(self,  gate, experts_, distribution_grid, is_encoder, options)

    def forward(self, input, **kwargs) -> Tensor:
        assert len(input.shape) == 3, "input Tensor must have dimensions: (s)equence, (t)oken, (m)odel"
        max_len = self.enc_max_len if self.is_encoder else self.dec_max_len
        d_model = input.shape[2]
        local_reshaped_input = input.reshape(-1, d_model)
        c_cpu = local_reshaped_input.shape[0]
        #add the allreduce to get the max_len
        self.get_max_len(c_cpu, max_len, local_reshaped_input.get_device())
        assert max_len["max_len"] >= c_cpu

        if self.expert_group_size > 1:
            reshaped_input_pad = torch.nn.functional.pad(local_reshaped_input, (0, 0, 0, max_len["max_len"] - c_cpu), 'constant', 0.0) #TODO: Add comment here
            reshaped_input = AllGather.apply(self.expert_group, reshaped_input_pad, 0)
        else:
            reshaped_input = local_reshaped_input


        # Nonpadding masks of the input tensor with original shape [s, b].
        # In top1gating, only nonpadding tokens are dispatched to experts.
        reshaped_nonpadding = kwargs['nonpadding'].reshape(-1) if kwargs.get('nonpadding', None) is not None else None

        #If reshaped_nonpadding is not None, need a Allgather to collect local nonpadding to generate the global one
        if self.expert_group_size > 1 and reshaped_nonpadding is not None:
            reshaped_nonpadding = torch.nn.functional.pad(reshaped_nonpadding, (0, max_len["max_len"] - c_cpu), 'constant', 0.0)
            reshaped_nonpadding = AllGather.apply(self.expert_group, reshaped_nonpadding, 0)

        combine_weights, dispatch_mask, expert_cumsum = self.gate(reshaped_input, nonpadding=reshaped_nonpadding)

        dispatched_input = reshaped_input[dispatch_mask % reshaped_input.shape[0]] #[sum*(E*C), M]

        if self._options.fp16_mode():
            dispatched_input = dispatched_input.to(torch.float16)

        expert_outputs = []
        if self.is_mergedFFNExpert:
            #TODO: This will be supported after optimized kernel. @Weixing Zhang
            assert 0, "unsupport merged FFN kernel for Expert Slicing"
        else:
            chunks = []
            for lb in range(len(expert_cumsum)-1):
                chunks.append(dispatched_input[expert_cumsum[lb] : expert_cumsum[lb+1]])
            assert len(chunks) == self.num_experts, f"len(chunks) is {len(chunks)}, num_experts is {self.num_experts}"

            for chunk, expert in zip(chunks, self.moe_experts):
                expert_outputs += [(expert(chunk))]
            expert_outputs = torch.cat(expert_outputs, dim=0)

        if self.expert_group_size > 1:
            expert_outputs = AllReduce.apply(expert_outputs, self.expert_group)

        #reshape back
        if reshaped_nonpadding is not None:
            reshaped_nonpadding_cumsum = reshaped_nonpadding.cumsum(dim=0) 
            reshaped_nonpadding_cumsum *= reshaped_nonpadding
            expert_outputs = torch.nn.functional.pad(expert_outputs, (0, 0, 1, 0), 'constant', 0.0) #does not matter which to put
            expert_outputs = expert_outputs.index_select(0, reshaped_nonpadding_cumsum)

        reshaped_input_pad = torch.nn.functional.pad(local_reshaped_input, (0, 0, 0, max_len["max_len"] - c_cpu), 'constant', 0.0)
        rerouted_output = torch.zeros(reshaped_input.shape[0] * self.gate.k, reshaped_input.shape[1], dtype = expert_outputs.dtype, device=expert_outputs.device) #dim [S*K, M]
        rerouted_output[dispatch_mask] = expert_outputs

        rerouted_output = rerouted_output.reshape(self.gate.k, reshaped_input.shape[0], reshaped_input.shape[1]) #reshaped to [K, S, M]

        #in general, combined_output = sum_i(combined[:, i]*rerouted_output[i,:,:])
        combined_output = om_einsum("ks,ksm->sm", combine_weights, rerouted_output.to(combine_weights))
        combined_output = combined_output.reshape(self.expert_group_size, -1, d_model)
        local_combined_output = torch.narrow(combined_output[self.expert_rank], dim = 0, start = 0, length = c_cpu)
        return local_combined_output.reshape(input.shape).to(input)


class MixtureOfExpertsEP(MixtureOfExperts):
    def __init__(self, gate, experts_, distribution_grid = None, is_encoder = True, options = {}):
        r"""MixtureOfExpertsEP module implements mixture of experts with expert parallelsim
        Args:
            gate: the gating function (required).
            experts_: list of experts (required).
            distribution_grid: DistributionGrid object providing interface to query torch.distributed process groups.
                It is a required keyword to remove the confusion for the usage of dgrid, for single GPU, instantiate an empty dgrid()
            is_encoder: Whether this MOE is in encoder layer. If false, it is decoder layer
        """
        MixtureOfExperts.__init__(self,  gate, experts_, distribution_grid, is_encoder, options)

    def forward(self, input:Tensor, **kwargs:Any) -> Tensor:
        assert len(input.shape) == 3, "input Tensor must have dimensions: (s)equence, (t)oken, (m)odel"
        max_len = self.enc_max_len if self.is_encoder else self.dec_max_len
        # Implement Algorithm 2 from GShard paper.
        d_model = input.shape[2]
        # Reshape into S tokens by dropping sequence dimension.
        reshaped_input = input.reshape(-1, d_model)

        if kwargs.get('shuffle_group', None) is not None and self._options.enable_base_layer_shuffling():
            shuffle_group = kwargs['shuffle_group']
        else:
            if kwargs.get('shuffle_group', None) is not None:
                print("WARNING: When Base Layer Shuffling is False, shuffle_group should not be set.")
            shuffle_group = self.expert_group

        if self._options.enable_base_layer_shuffling():
            S = reshaped_input.shape[0]
            perm = torch.randperm(S)
            reshaped_input = reshaped_input[perm]
            reshaped_input = reshaped_input.reshape(self.expert_group_size, -1, d_model) # e x s/e x m
            reshaped_input = AllToAll.apply(shuffle_group, reshaped_input, reshaped_input.shape[-2]) # e x s/e x m
            reshaped_input = reshaped_input.reshape(-1, d_model) # s x m

        # Nonpadding masks of the input tensor with original shape [s, t].
        # In top1gating, only nonpadding tokens are dispatched to experts.
        # lid is the layer id for warning message. Default value -1 will not trigger the warning.
        reshaped_nonpadding = kwargs['nonpadding'].reshape(-1) if kwargs.get('nonpadding', None) is not None else None
        combine_weights, dispatch_mask, capacity_fp = self.gate(reshaped_input, nonpadding=reshaped_nonpadding, lid=kwargs.get('lid', -1))
        dispatched_input = reshaped_input.index_select(0, (dispatch_mask % reshaped_input.shape[0]).reshape(-1)).reshape(-1, math.ceil(capacity_fp), d_model)
        # index_select() is used to replace advance indexing because dispatch_mask has duplicate indices.
        # Backward pass is slow when advance indexing contains duplicate indices

        if self._options.fp16_mode() and dispatched_input.dtype is not torch.float16:
            dispatched_input = dispatched_input.to(torch.float16)
        c_cpu = dispatched_input.shape[1]
        if self.expert_group_size > 1:
            if not math.isclose(capacity_fp,max_len["capacity_fp"]):
                max_len["capacity_fp"] = capacity_fp
                max_len["need_update"] = True
            #add the allreduce to get the max_len
            self.get_max_len(c_cpu, max_len, dispatched_input.get_device())
            assert max_len["max_len"] >= c_cpu
            dispatched_input = AllToAll.apply(shuffle_group, dispatched_input, max_len['max_len'])
        # Re-shape after all-to-all: ecm -> gecm
        dispatched_input = dispatched_input.reshape(self.expert_group_size, self.num_experts, -1, d_model)

        if not self.is_mergedFFNExpert:
            chunks = dispatched_input.chunk(self.num_experts, dim=1)
            expert_outputs = []
            for chunk, expert, e in zip(chunks, self.moe_experts, range(self.num_experts)):
                if self._options.checkpoint_experts() is True:
                    expert_outputs += [checkpoint.checkpoint(expert, chunk)]
                elif self._options.enable_expert_weight_calculation_optimization() is True:
                    # Only process input chunk for the selected experts. The input chunk is padded upto the capacity factor
                    z = torch.min(dispatch_mask[e], dim=0)
                    truncated_chunk = None
                    if z.values == -1:
                        truncated_chunk = chunk[...,:z.indices,:]
                        eo = expert(truncated_chunk)
                        pad_size = math.ceil(capacity_fp) - truncated_chunk.size(dim=2)
                        expert_outputs +=[torch.nn.functional.pad(eo, (0,0,pad_size,0), 'constant', 0.0)]  
                    else:
                        expert_outputs += [expert(chunk)]
                else:
                    expert_outputs += [expert(chunk)]
            expert_output = torch.cat(expert_outputs, dim=1)
        else:
            expert_output = self.moe_experts(dispatched_input)

        if self.expert_group_size > 1:
            expert_output = AllToAll.apply(shuffle_group, expert_output, max_len['max_len'])
        expert_output = torch.narrow(expert_output, dim = 2, start=0, length = c_cpu)
        # Re-shape back: gecm -> ecm
        expert_output = expert_output.reshape(self.expert_group_size * self.num_experts, -1, d_model)

        rerouted_output = torch.zeros(reshaped_input.shape[0] * self.gate.k+1, reshaped_input.shape[1], dtype = expert_output.dtype, device=expert_output.device) #dim [S*K, M]
        rerouted_output[dispatch_mask] = expert_output
        rerouted_output = rerouted_output[0:-1]
        rerouted_output = rerouted_output.reshape(self.gate.k, reshaped_input.shape[0], reshaped_input.shape[1]) #reshaped to [K, S, M]

        #in general, combined_output = sum_i(combined[:, i]*rerouted_output[i,:,:])
        combined_output = om_einsum("ks,ksm->sm", combine_weights, rerouted_output.to(combine_weights))

        if self._options.enable_base_layer_shuffling():
            combined_output = combined_output.reshape(self.expert_group_size, -1, d_model) # e x s/e x m
            combined_output = AllToAll.apply(shuffle_group, combined_output, combined_output.shape[-2]) # e x s/e x m
            combined_output = combined_output.reshape(-1, d_model) # s x m
            inverse_shuffle_output = torch.empty_like(combined_output)
            inverse_shuffle_output[perm] = combined_output
            combined_output = inverse_shuffle_output.reshape(input.shape).to(input)
        else:
            combined_output = combined_output.reshape(input.shape).to(input)

        return combined_output.reshape(input.shape).to(input)
