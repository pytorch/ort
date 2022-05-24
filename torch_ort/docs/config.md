## How to configure torch-ort

### Log level

### Save ONNX files

```python
from torch_ort import DebugOptions
model = torch_ort.ORTModule(TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout), DebugOptions(save_onnx=True, onnx_prefix='TransformerModel')).to(device)
```

### Or load from config

```python
import os
from torch_ort.experimental.json_config import load_from_json
path_to_json = os.path.join(os.getcwd(), 'torch-ort_config.json')
load_from_json(model, path_to_json)
```

```json
{
  "DebugOptions": {
    "SaveONNX": true,
    "ONNXPrefix": "torch-ort_model"
  }
}
```

## Access the original torch module

`._torch_module`