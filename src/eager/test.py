import torch
import torch_ort

device = torch.device("ort")
x = torch.empty(2, 3, device = device)
y = torch.empty(2, 3, device = device)
z = y + x
z2 = x.cpu() + y.cpu()
print(z.size())
print(z.cpu())
print(z2)
