# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch
import torch_ort

device = torch_ort.device.cpu()

ones = torch.ones(2, 3).to(device)
print(ones.cpu())

twos = ones + ones
print(twos.cpu())

threes = twos + ones
print(threes.cpu())

fours = twos * twos
print(fours.cpu())

fenced_ten = torch.tensor(
  [[-1, -1, -1],
   [-1, 10, -1],
   [-1, -1, -1]],
  device = device, dtype=torch.float)

print(fenced_ten.numel())
print(fenced_ten.size())
print(fenced_ten.cpu())
print(fenced_ten.relu().cpu())

a = torch.ones(3, 3).to(device)
b = torch.ones(3, 3)
c = a + b
print(c.cpu())