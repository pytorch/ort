# This tutorial is based on PyTorch Langguage modeling tutorial https://pytorch.org/tutorials/beginner/transformer_tutorial.html


import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import dataset
import sys
import os
from typing import Tuple

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from ort_moe.layers import TransformerMoEEncoderLayer
from ort_moe.topKgate import TopKGate
from ort_moe.grids import DistributionGrid

# Step 1 : Define the MoE Model
# Here we build a transformer model, SimpleMoE, where every other Transformer encder layer is a MOE layer.

class SimpleMoE(nn.Module):
    """SimpleMoE is a n layer Gshard style encoders only MoE model. Every other layer is mixer of experts.
    """
    def __init__(self, ntokens : int,  d_model : int,  nheads : int, nhid : int,  nlayers : int, nexperts : int, dropout: float, dg):
        super().__init__()
        assert nlayers % 2 == 0, "Only even numbers of layers are supported"
        self.src_mask = None
        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        self.encoders = nn.ModuleList()
        
        for i in range (nlayers):

            # Add standard transformer encoder layer
            self.encoders.append(nn.TransformerEncoderLayer(d_model, nheads, nhid))

            # Add MoE encoder layer
            # Set up the gating functino for the mixsture of experts
            top2gate = TopKGate(d_model, nexperts, dgrid=dg)
            self.encoders.append(TransformerMoEEncoderLayer(d_model, nheads, nhid, dropout,
                                                            nexperts = nexperts, gate = top2gate,
                                                            distribution_grid = dg))
            i = i + 1

        self.embedding = nn.Embedding(ntokens, d_model)
        self.decoder = nn.Linear(d_model, ntokens)

        initrange = 0.1
        nn.init.uniform_(self.embedding.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        src = self.embedding(src) * math.sqrt(self.d_model)
        output = self.pos_encoder(src)
        for me in self.encoders:
            output = me(output)
        output = self.decoder(output)
        return F.log_softmax(output, dim=-1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# Step 2 : Setup and load data. 
# This is a stardard wikitest data loading routine which has no MoE specific changes.

from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

def batchify(data: Tensor, bsz: int) -> Tensor:
    """Divides the data into bsz separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Args:
        data: Tensor, shape [N]
        bsz: int, batch size

    Returns:
        Tensor of shape [N // bsz, bsz]
    """
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data

def load_data(batch_size, eval_batch_size):
    train_iter = WikiText2(split='train')
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>']) 

    def data_process(raw_text_iter: dataset.IterableDataset) -> Tensor:
        """Converts raw text into a flat Tensor."""
        data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    # train_iter was "consumed" by the process of building the vocab,
    # so we have to create it again
    train_iter, val_iter, test_iter = WikiText2()
    train_data = data_process(train_iter)
    val_data = data_process(val_iter)
    test_data = data_process(test_iter)

    train_data = batchify(train_data, batch_size).to(local_rank)  # shape [seq_len, batch_size]
    val_data = batchify(val_data, eval_batch_size).to(local_rank)
    test_data = batchify(test_data, eval_batch_size).to(local_rank)

    return train_data, val_data, test_data, vocab

def get_batch(source: Tensor, i: int) -> Tuple[Tensor, Tensor]:
    """
    Args:
        source: Tensor, shape [full_seq_len, batch_size]
        i: int

    Returns:
        tuple (data, target), where data has shape [seq_len, batch_size] and
        target has shape [seq_len * batch_size]
    """
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target


# Step 3 : Define training and evaluation loop
# This step is not MoE specific

import copy
import time

criterion = nn.CrossEntropyLoss()
lr = 5.0  # learning rate

def train(model: nn.Module, optimizer, scheduler, epoch) -> None:
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 200
    start_time = time.time()

    num_batches = len(train_data) // bptt
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        batch_size = data.size(0)
        output = model(data)
        loss = criterion(output.view(-1, ntokens), targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()
    
    # Reduce experts in Expert Parallel Replica mode
    from ort_moe.utils import moe_module_all_reduce_experts
    moe_module_all_reduce_experts(model, dg)

def evaluate(model: nn.Module, eval_data: Tensor) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, bptt):
            data, targets = get_batch(eval_data, i)
            batch_size = data.size(0)
            output = model(data)
            output_flat = output.view(-1, ntokens)
            total_loss += batch_size * criterion(output_flat, targets).item()
    return total_loss / (len(eval_data) - 1)

def run_model(model):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(model, optimizer, scheduler, epoch)
        val_loss = evaluate(model, val_data)
        val_ppl = math.exp(val_loss)
        elapsed = time.time() - epoch_start_time
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
              f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
        print('-' * 89)

        scheduler.step()

# Step 4 Instantiate the model

# set training parameters
epochs = 1
batch_size = 20
eval_batch_size = 10
bptt = 35
local_rank = 0
if os.environ.get("LOCAL_RANK") is not None:
    local_rank = int(os.environ["LOCAL_RANK"])

# load data
train_data, val_data, test_data, vocab = load_data(batch_size, eval_batch_size)

# set model parameters
ntokens = len(vocab)  # size of vocabulary
emsize = 200  # embedding dimension
d_hid = 200  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2  # number of heads in nn.MultiheadAttention
nexperts = 8 # number of experts in MoE layer
dropout = 0.2  # dropout probability

# Step 4 Run the model

# There are many configurations we can explore. 
# Enable one of the following 4 configurations.

# Configuration 1 : Run the model on single GPU
# ---------------------------------------------
dg = DistributionGrid()
model = SimpleMoE(ntokens, emsize, nhead, d_hid, nlayers, nexperts, dropout, dg).to(local_rank)
run_model(model)

# Configuration 2 : Run the model in Data Parallel mode on 4 GPUs
# ---------------------------------------------------------------
# Create distribution grid to manage process groups
#dg = DistributionGrid(data_parallel_group_size = 4)
#model = SimpleMoE(ntokens, emsize, nhead, d_hid, nlayers, nexperts, dropout, dg).to(local_rank)
#
# Use torch DistributedDataParallel to do Data Parallel
#pg = dg.get_data_parallel_group()
#model = DDP(model, device_ids=[local_rank], output_device=local_rank, process_group=pg)
#
#run_model(model)
# dg.cleanup()

# Configuration 3 : Run the model in Expert Parallel mode on 4 GPUs
# -----------------------------------------------------------------
# Create distribution grid to manage process groups
#dg = DistributionGrid(expert_parallel_group_size = 4)
#model = SimpleMoE(ntokens, emsize, nhead, d_hid, nlayers, nexperts, dropout, dg).to(local_rank)

# Use torch DistributedDataParallel to do data parallel for non-expert parameters only
#pg = dg.get_data_parallel_group()
#from ort_moe.utils import exclude_moe_params_in_ddp
#exclude_moe_params_in_ddp(model)

#model = DDP(model, device_ids=[local_rank], output_device=local_rank, process_group=pg)
#run_model(model)
#dg.cleanup()

# Configuration 4 : Run Expert Parallel Replicas.
# -----------------------------------------------
# Here 2 replicas of the model in Expert Parallel mode (4 GPUs each) is run using total 8 GPUs.
# Create distribution grid to manage process groups
#dg = DistributionGrid(expert_parallel_group_size = 4, expert_parallel_replica_group_size = 2)
#model = SimpleMoE(ntokens, emsize, nhead, d_hid, nlayers, nexperts, dropout, dg).to(local_rank)

# Use torch DistributedDataParallel to do data parallel for non-expert parameters only
#pg = dg.get_data_parallel_group()
#from ort_moe.utils import exclude_moe_params_in_ddp
#exclude_moe_params_in_ddp(model)

#model = DDP(model, device_ids=[local_rank], output_device=local_rank, process_group=pg)
# Expert parameters between two replicas should be reduced after processing the batch using moe_module_all_reduce_experts utility.
#run_model(model)
#dg.cleanup()

