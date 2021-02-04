import torch
import torch_ort

#todo: move it to torch_ort as util function
def get_ort_device(devkind, index = 0):
    return torch.device("ort", torch_ort.get_ort_device(devkind, index))

device = get_ort_device("Apollo")
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

