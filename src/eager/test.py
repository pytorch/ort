import torch
import torch_ort

device = torch_ort.device.apollo()

x = torch.empty(2, 3, device = device)
y = torch.empty(2, 3, device = device)
z = y + x
z2 = x.cpu() + y.cpu()
print(z.size())
print(z.cpu())
print(z2)

s = x - y
print(s.cpu())

m = x * y
print(m.cpu())

r = torch.relu(x)
print(r.cpu())

