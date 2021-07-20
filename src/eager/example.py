import torch
import torch_ort

torch_ort.set_device(0, 'CPUExecutionProvider', {})
#torch_ort.set_device(1, 'NPUExecutionProvider', {'shared_lib_path': 'provider.so', 'device_id':'0'})
ort_device = torch_ort.device(0)
torch.zeros(10, 10, device=ort_device)
