import torch
import torch_ort

device = torch.device("ort")
x = torch.empty(5, 3, device = device)
print(x.size())
print(x)