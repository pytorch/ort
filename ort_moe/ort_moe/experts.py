# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class FFNExpert(nn.Module):
    r"""FFNExpert is a simple feedforward network aka as expert.

    Args:
        d_model(int): the number of expected features in the input (required).
        dim_feedforward(int): the dimension of the feedforward network model (required).
        dgrid(DistributionGrid): DistributionGrid object providing interface to query torch.distributed process groups (required)
            It is a required keyword to remove the confusion for the usage of dgrid, for single GPU, instantiate an empty dgrid()
        activation_fn(nn.functional): The activation function for expert compuation, default is ReLU
        expert_dropout(float): the dropout rate for the dropout layer inside expert, may need a large number, say 0.4, for finetuning tasks.
    """
    def __init__(self, d_model, dim_feedforward, dgrid, activation_fn = nn.functional.relu, expert_dropout = 0.0):
        super().__init__()
        self.mp_size = 1
        if dgrid is not None:
            self.mp_size = dgrid.get_expert_slicing_world_size()
        self.linear1 = nn.Linear(d_model, dim_feedforward//self.mp_size, bias=False)
        self.linear2 = nn.Linear(dim_feedforward//self.mp_size, d_model, bias=False)
        self.activation_fn = activation_fn
        self.expert_dropout_rate = expert_dropout

    def forward(self, x: torch.tensor):
        x = self.linear1(x.float())
        x = self.activation_fn(x)
        if self.expert_dropout_rate > 0:
            x = F.dropout(x, p=self.expert_dropout_rate, training=self.training)
        x = self.linear2(x.float())
        return x

class MergedFFNExpert(nn.Module):
    r"""FFNExpert is a simple feedforward network aka as expert.

    Args:
        d_model(int): the number of expected features in the input (required).
        dim_feedforward(int): the dimension of the feedforward network model (required).
        local_num_experts(int): The number of experts on current device
        dgrid(DistributionGrid): DistributionGrid object providing interface to query torch.distributed process groups (required)
            It is a required keyword to remove the confusion for the usage of dgrid, for single GPU, instantiate an empty dgrid()
        activation_fn(nn.functional): The activation function for expert compuation, default is ReLU
        expert_dropout(float): the dropout rate for the dropout layer inside expert, may need a large number, say 0.4, for finetuning tasks.
    """

    def __init__(self, d_model, dim_feedforward, local_num_experts, dgrid, activation_fn = nn.functional.relu, expert_dropout = 0.0):
        super().__init__()
        self.mp_size = dgrid.get_expert_slicing_world_size()

        self.weight1 = nn.Parameter(torch.Tensor(local_num_experts, d_model, dim_feedforward//self.mp_size)) #emf
        self.weight2 = nn.Parameter(torch.Tensor(local_num_experts, dim_feedforward//self.mp_size, d_model)) #efm

        with torch.no_grad():
            # make initialization the same with FFNExpert
            for i in range(local_num_experts):
                wshape = self.weight1[i].shape
                nn.init.kaiming_uniform_(self.weight1[i].view(wshape[1], wshape[0]), a=math.sqrt(5))
                self.weight1[i] = self.weight1[i].view(wshape[1], wshape[0]).t().detach().clone()
                wshape = self.weight2[i].shape
                nn.init.kaiming_uniform_(self.weight2[i].view(wshape[1], wshape[0]), a=math.sqrt(5))
                self.weight2[i] = self.weight2[i].view(wshape[1], wshape[0]).t().detach().clone()
        self.activation_fn = activation_fn
        self.local_num_experts = local_num_experts
        self.expert_dropout_rate = expert_dropout

    def forward(self, x: torch.tensor):
        x = x.transpose(0, 1) #gecm --> egcm
        input_shape = x.shape
        reshaped_x = x.reshape(input_shape[0], -1, input_shape[-1]) #egcm --> e,gxc,m
        out1 = torch.bmm(reshaped_x.float(), self.weight1) #e, gxc, f
        out1 = self.activation_fn(out1)
        if self.expert_dropout_rate > 0:
            out1 = F.dropout(out1, p=self.expert_dropout_rate, training=self.training)
        out2 = torch.bmm(out1.float(), self.weight2) #e, gxc, m
        out2 = out2.reshape(input_shape)
        out2 = out2.transpose(0, 1) #egcm --> gecm
        return out2