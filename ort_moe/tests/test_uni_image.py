# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
########################
# ORTModule related tests
########################

#a simple test to make sure ortmodule result is the same as pytorch result
from torch_ort import ORTModule
import torch
class M(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.linear = torch.nn.Linear(4,4)

  def forward(self, x):
    y = self.linear(x)
    return y

model = M()
model_ort = ORTModule(model)

input = torch.rand(4, 4).requires_grad_()
output = model(input)
output_ort = model_ort(input)
assert torch.allclose(output, output_ort)

########################
# SCCL/NCCL Related test
########################
#TODO
