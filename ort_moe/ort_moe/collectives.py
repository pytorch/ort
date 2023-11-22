# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import torch
import torch.distributed as dist

class AllGather(torch.autograd.Function):
    """
    Gather the tensors from the whole group into list,
    which is concatinated into a single tensor
    """
    @staticmethod
    def forward(ctx, group, input, gather_dim):
        """
        group: The communication group
        input: The input tensor to be gathered
        gather_dim: The dimension of the input tensor the the gathered tensors should be concatinated on
        """
        ctx.group = group
        ctx.group_size = dist.get_world_size(group)
        ctx.gather_dim = gather_dim
        ctx.rank = torch.distributed.get_rank(group=ctx.group)

        out_tensor_list = [torch.empty_like(input) for i in range(ctx.group_size)]

        dist.all_gather(out_tensor_list, input, group=group)
        out_tensor = torch.cat(out_tensor_list, dim=gather_dim)
        return out_tensor

    @staticmethod
    def backward(ctx, *grad_output):
        """
        The AD for AllGather is a reduction op,
        The gradients are chunked into a list based on the group size and gather dimension,
        Only the i-th chunk is returned, where i is the current rank
        """
        # ORT supports only contiguous tensors for now.
        # Once the issue is fixed .contiguous() can be removed

        dist.all_reduce(*grad_output, group = ctx.group)
        grad_output_chunks = (grad_output[0]).chunk(ctx.group_size, dim = ctx.gather_dim)
        output = (grad_output_chunks[ctx.rank]).contiguous() 
        return None, output, None

def compressed_all_to_all(output, input, group=None):

    world_size = dist.get_world_size(group)
    rank = torch.distributed.get_rank(group)

    ts_in = torch.tensor_split(input, world_size)
    compressed_a2a_input, _ = dg_compress(ts_in)
    ts_out = torch.tensor_split(output, world_size)

    for i in range(world_size):
        if i != rank:
            torch.distributed.send(compressed_a2a_input[i], i)
            torch.distributed.recv(compressed_a2a_input[i], i)
    dg_decompress(compressed_a2a_input, ts_out)

# Based on https://github.com/pytorch/pytorch/pull/40762
class AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, input, max_len_cpu):  # type: ignore
        """
        The Alltoall on the input tensor of a group of ranks
        Args:
            group(torch.distributed.ProcessGroup): The communication group of alltoall
            input (torch.Tensor): The input tensor
            max_len_cpu (int): The maximum length the tensor should be pad to on the second last dimension of the input tensor. This is on CPU
        """
        ctx.group = group
        ctx.input_c = input.shape[-2]
        ctx.max_len_cpu = max_len_cpu

        #pad the input to the max_len
        assert max_len_cpu >= input.shape[-2]
        if max_len_cpu != input.shape[-2]:
            input = torch.nn.functional.pad(input, pad=(0,0,0,max_len_cpu-input.shape[-2]), mode = 'constant', value=0.0)
        input = input.contiguous()
        output = torch.empty_like(input)
        if os.environ.get('ORT_MOE_COMPRESS_ALLToALL') is not None:
            compressed_all_to_all(output, input, group=group)
        else:
            dist.all_to_all_single(output, input, group=group)
        return output

    @staticmethod
    def backward(ctx, *grad_output):
        input_c = ctx.input_c
        assert ctx.max_len_cpu >= input_c
        ret_val = AllToAll.apply(ctx.group, *grad_output, ctx.max_len_cpu)
        narrow_dim = len(ret_val.shape)-2
        # ORT supports only contiguous tensors for now.
        # Once the issue is fixed .contiguous() can be removed
        ret_val = torch.narrow(ret_val, dim = narrow_dim, start=0, length = input_c).contiguous()
        return None, ret_val, None

class AllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, group):
        """
        The allreduce on the input tensor on a group of ranks
        Args:
            group(torch.distributed.ProcessGroup): The communication group 
            input (torch.Tensor): The input tensor to be all-reduced
        """
        ctx.group = group
        dist.all_reduce(input, group=group)
        return input

    @staticmethod
    def backward(ctx, *grad_output):
        """
        The AD of Allreduce is Allreduce
        """
        # ORT supports only contiguous tensors for now.
        # Once the issue is fixed .contiguous() can be removed
        dist.all_reduce(grad_output[0], group = ctx.group)
        ret_val = grad_output[0].contiguous()
        return ret_val, None

_DIETGPU_SCRATCH_PAD = None
_DIETGPU_LIBRARY_LOADED = False
def dg_load_library():
    global _DIETGPU_LIBRARY_LOADED
    if _DIETGPU_LIBRARY_LOADED is True:
        return

    dg_lib = os.environ.get('DIETGPU_LIB_PATH')
    torch.ops.load_library(dg_lib)
    _DIETGPU_LIBRARY_LOADED = True
    return

def get_tensor_list_device(the_list):
    if isinstance(the_list, torch.Tensor):
        return the_list.device

    for l in the_list:
        if isinstance(l, torch.Tensor):
            return l.device

    return None

def dg_get_scratch_pad(device):
    global _DIETGPU_SCRATCH_PAD
    if _DIETGPU_SCRATCH_PAD is None:
        _DIETGPU_SCRATCH_PAD = torch.empty([64*1024*1024], dtype=torch.uint8, device=device)
    return _DIETGPU_SCRATCH_PAD

def dg_compress(input_list):
    dg_load_library()

    output, output_size, _ = torch.ops.dietgpu.compress_data(True, input_list, dg_get_scratch_pad(get_tensor_list_device(input_list)))
    compressed_output_list = []
    for size, t in zip(output_size, [*output]):
        truncated_t = t.narrow(0, 0, size.item()).clone()
        compressed_output_list.append(truncated_t)

    return compressed_output_list, output_size

def dg_decompress(input_list, output_list):
    dg_load_library()
    torch.ops.dietgpu.decompress_data(True, input_list, output_list, dg_get_scratch_pad(get_tensor_list_device(input_list)))
