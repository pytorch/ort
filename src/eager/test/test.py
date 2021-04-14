import torch
import torch_ort

cpu_ones = torch.Tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
ort_ones = cpu_ones.to(torch_ort.device.cpu())
torch.sin_(ort_ones)