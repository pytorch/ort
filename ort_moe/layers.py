# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from torch import Tensor
import torch.nn as nn

from typing import Optional

from ort_moe.experts import FFNExpert, MergedFFNExpert
from ort_moe.topKgate import TopKGate
from ort_moe.moe import MixtureOfExpertsFunc
from ort_moe.utils import fsdp_wrap

class TransformerMoEEncoderLayer(nn.Module):
    r"""TransformerMoEEncoderLayer is made up of muti headded attention, and gated collection
    of feedforward networks aka experts.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        nexprts: the number of experts(default=64).
        balance_ratio: The scaling ratio for the loss_aux
        gate: the gating function (default=top2k).
        fp16_mode : True if FP16 mode is enabled. Default is 'False'
        expertslist: List of experts of type nn.ModuleList.
        merged_expert: Whether the experts are mergedFFN experts
        distribution_grid: DistributionGrid object providing interface to query torch.distributed process groups
        use_fsdp : Use FullyShardedDataParallel to shard the layer. Default is 'False'
        flatten_parameters : Flatten sharded paratmers when use_fsdp is True. Default is 'True'
        apex_opt_level : Default 'None'

    Examples::
        >>> moe_layer = nn.TransformerMoEEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = moe_layer(src)
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 nexperts=64, balance_ratio = 0.01, gate=None, fp16_mode=False,
                 expertslist=None, merged_expert=True, use_mpi4py=False, distribution_grid=None,
                 use_fsdp=False, flatten_parameters=True, apex_opt_level=None):
        super(TransformerMoEEncoderLayer, self).__init__()

        if not gate: #default is top1
            gate = TopKGate(d_model, nexperts, balance_ratio=balance_ratio, fp16_mode = fp16_mode, k = 1, dgrid=distribution_grid)
        # attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # experts
        if distribution_grid.get_expert_parallel_group() is not None:
            self.n_local_experts = nexperts // distribution_grid.get_expert_parallel_world_size() 
        else:
            self.n_local_experts = nexperts

        if not merged_expert:
            if expertslist is None:
                experts = nn.ModuleList()

                for i in range(self.n_local_experts):
                    e = FFNExpert(d_model, dim_feedforward, dgrid=distribution_grid)
                    experts.append(e)
            else:
                experts = expertslist
                self.n_local_experts = len(experts)
        else:
            experts = MergedFFNExpert(d_model, dim_feedforward, self.n_local_experts, dgrid=distribution_grid)

        # mixer of experts
        self.moe = MixtureOfExpertsFunc(gate, experts, is_encoder=True,
                                    fp16_mode=fp16_mode, use_mpi4py=use_mpi4py, distribution_grid=distribution_grid)

        if use_fsdp is True:
            mp = False
            if apex_opt_level == "O2":
                mp = True 
            fsdp_config = dict(mixed_precision=mp, process_group=distribution_grid.get_moe_group())
            if flatten_parameters is False:
                fsdp_config['flatten_parameters'] = False
            self.moe = fsdp_wrap(self.moe, **fsdp_config)

        # drop and norm layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        #Add and norm
        src = src + self.dropout1(src2) #Dropout1 seems not in the Gshard figure 3, I assume it is omitted
        src = self.norm1(src)
        src2 = self.moe(src)
        src = src + self.dropout2(src2) #Dropout2 seems not in the Gshard figure 3, I assume it is omitted
        src = self.norm2(src)

        return src

class LanguageExpertMoEEncoderLayer(nn.Module):
    r"""LanguageExpertMoEEncoderLayer is made up of muti headded attention, and gated collection
    of feedforward networks aka experts./asse

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        nexprts: the number of experts(default=64).
        balance_ratio: The scaling ratio for the loss_aux
        gate: the gating function (default=top2k).
        fp16_mode : True if FP16 mode is enabled. Default is 'False'
        expertslist: List of experts of type nn.ModuleList.
        nlang_experts: number of language experts
        use_mpi4py: Whether use mpi4py library or nccl package
        distribution_grid: DistributionGrid object providing interface to query torch.distributed process groups
        merged_expert: whether the MoE experts is MergedFFNExpert
        use_fsdp : Use FullyShardedDataParallel to shard the layer
        flatten_parameters : Flatten sharded paratmers when use_fsdp is True. Default is 'True'
        apex_opt_level : Default 'None'

    Examples::
        >>> moe_layer = nn.LanguageExpertMoEEncoderLayer(d_model=512, nhead=8, nlang_experts=4)
        >>> src = torch.rand(10, 32, 512)
        >>> out = moe_layer(src)
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 nexperts=64, balance_ratio = 0.01, gate=None, fp16_mode=False,
                 expertslist=None, nlang_experts=4, use_mpi4py=False, distribution_grid=None, merged_expert=True,
                 use_fsdp=False, flatten_parameters=True, apex_opt_level=None):
        super(LanguageExpertMoEEncoderLayer, self).__init__()
        self.experts = nn.ModuleDict()
        for i in range(nlang_experts):
            le = TransformerMoEEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                        nexperts = nexperts, balance_ratio = balance_ratio,
                                                        gate = gate, fp16_mode = fp16_mode,
                                                        expertslist = expertslist, use_mpi4py=use_mpi4py,
                                                        distribution_grid=distribution_grid, 
                                                        merged_expert=merged_expert, use_fsdp=use_fsdp
                                                        )
            self.experts[f"seq2seq{i}"] = le

    def forward(self, src, lang_id=None):
        expert = self.experts[f"seq2seq{lang_id}"]
        return expert(src)

class TransformerMoEDecoderLayer(nn.Module):
    r"""TransformerMoEDecoderLayer is made up of muti headded attention, and gated collection
    of feedforward networks aka experts.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        nexprts: the number of experts(default=64).
        balance_ratio: The scaling ratio for the loss_aux
        gate: the gating function (default=None. If none then top2 gating is used).
        fp16_mode : True if FP16 mode is enabled. Default is 'False'
        expertslist: List of experts of type nn.ModuleList.
        merged_expert: whether the MoE experts is MergedFFNExpert
        use_mpi4py: Whether use mpi4py library or nccl package
        distribution_grid: DistributionGrid object providing interface to query torch.distributed process groups
        use_fsdp : Use FullyShardedDataParallel to shard the layer
        flatten_parameters : Flatten sharded paratmers when use_fsdp is True. Default is 'True'
        apex_opt_level : Default 'None'

    Examples::
        >>> moe_layer = nn.TransformerMoEDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> src = torch.rand(20, 32, 512)
        >>> out = moe_layer(src, memory)
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 nexperts=64, balance_ratio = 0.01, gate=None, fp16_mode=False,
                 expertslist=None, merged_expert = True, use_mpi4py=False, distribution_grid=None,
                 use_fsdp=False, flatten_parameters=True, apex_opt_level=None):
        super(TransformerMoEDecoderLayer, self).__init__()

        # attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # experts
        if distribution_grid.get_expert_parallel_group() is not None:
            self.n_local_experts = nexperts // distribution_grid.get_expert_parallel_world_size() 
        else:
            self.n_local_experts = nexperts

        if not merged_expert:
            if expertslist is None:
                experts = nn.ModuleList()
                for i in range(self.n_local_experts):
                    e = FFNExpert(d_model, dim_feedforward, dgrid=distribution_grid)
                    experts.append(e)
            else:
                experts = expertslist
                self.n_local_experts = len(experts)
        else:
            experts = MergedFFNExpert(d_model, dim_feedforward, self.n_local_experts, dgrid=distribution_grid)

        # gate
        if not gate: #default is top2
            gate = TopKGate(d_model, nexperts, balance_ratio=balance_ratio, fp16_mode = fp16_mode, dgrid=distribution_grid)

        # mixer of experts
        self.moe = MixtureOfExpertsFunc(gate, experts, is_encoder=False,
                                        fp16_mode = fp16_mode, use_mpi4py = use_mpi4py, distribution_grid = distribution_grid)

        if use_fsdp is True:
            mp = False
            if apex_opt_level == "O2":
                mp = True 
            fsdp_config = dict(mixed_precision=mp, process_group=distribution_grid.get_moe_group())
            if flatten_parameters is False:
                fsdp_config['flatten_parameters'] = False
            self.moe = fsdp_wrap(self.moe, **fsdp_config)

        # drop and norm layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, tgt: Tensor, memory: Tensor,
                tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            tgt: the sequence to the encoder layer (required).
            tgt_mask: the mask for the tgt sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
        """
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        #Add and norm
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.moe(tgt)

        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt

class LanguageExpertMoEDecoderLayer(nn.Module):
    r"""LanguageExpertMoEDecoderLayer is made up of muti headded attention, and gated collection
    of feedforward networks aka experts.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        nexprts: the number of experts(default=64).
        balance_ratio: The scaling ratio for the loss_aux
        gate: the gating function (default=top2k).
        fp16_mode : True if FP16 mode is enabled. Default is 'False'
        expertslist: List of experts of type nn.ModuleList
        nlang_experts: number of language experts
        use_mpi4py: Whether use mpi4py library or nccl package
        distribution_grid: DistributionGrid object providing interface to query torch.distributed process groups
        merged_expert: whether the MoE experts is MergedFFNExpert
        use_fsdp : Use FullyShardedDataParallel to shard the layer
        flatten_parameters : Flatten sharded paratmers when use_fsdp is True. Default is 'True'
        apex_opt_level : Default 'None'


    Examples::
        >>> moe_layer = nn.LanguageExpertMoEDecoderLayer(d_model=512, nhead=8, nlang_experts=4)
        >>> src = torch.rand(10, 32, 512)
        >>> out = moe_layer(src)
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 nexperts=64, balance_ratio = 0.01, gate=None, fp16_mode=False,
                 expertslist=None, nlang_experts=4, use_mpi4py=False, distribution_grid=None, merged_expert=True,
                 use_fsdp=False, flatten_parameters=True, apex_opt_level=None):
        super(LanguageExpertMoEDecoderLayer, self).__init__()
        self.experts = nn.ModuleDict()
        for i in range(nlang_experts):
            le = TransformerMoEDecoderLayer(d_model, nhead, dim_feedforward, dropout,
                                            nexperts = nexperts, balance_ratio = balance_ratio,
                                            gate = gate, fp16_mode = fp16_mode,
                                            expertslist = expertslist, use_mpi4py=use_mpi4py,
                                            distribution_grid=distribution_grid, merged_expert=merged_expert, use_fsdp=use_fsdp)
            self.experts[f"seq2seq{i}"] = le

    def forward(self, tgt, memory, lang_id=None):
        expert = self.experts[f"seq2seq{lang_id}"]
        return expert(tgt, memory)
