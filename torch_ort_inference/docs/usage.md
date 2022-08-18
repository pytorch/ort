# APIs for OpenVINO™ integration with TorchORT

This document describes available Python APIs for OpenVINO™ integration with TorchORT to accelerate inference for PyTorch models on various Intel hardware.

## Essential APIs

To add the OpenVINO™ integration with TorchORT package to your PyTorch application, add following 2 lines of code:

```python
from torch_ort import ORTInferenceModule
model = ORTInferenceModule(model)
```

By default, CPU backend with FP32 precision is enabled. You can set different backend and supported precision using OpenVINOProviderOptions as below:

```python
from torch_ort import ORTInferenceModule, OpenVINOProviderOptions
provider_options = OpenVINOProviderOptions(backend = "GPU", precision = "FP16")
model = ORTInferenceModule(model, provider_options = provider_options)
```
Supported backend-precision combinations:
| Backend | Precision |
| --------| --------- |  
|   CPU   |    FP32   |
|   GPU   |    FP32   |
|   GPU   |    FP16   |
|  MYRIAD |    FP16   |

## Additional APIs

To save the inline exported onnx model, use DebugOptions as below:

```python
from torch_ort import ORTInferenceModule, DebugOptions
debug_options = DebugOptions(save_onnx=True, onnx_prefix='<model_name>')
model = ORTInferenceModule(model, debug_options=debug_options)
```

To enable verbose log of the execution of the TorchORT pipeline, use DebugOptions as below:

```python
from torch_ort import ORTInferenceModule, DebugOptions, LogLevel
debug_options = DebugOptions(log_level=LogLevel.VERBOSE)
model = ORTInferenceModule(model, debug_options=debug_options)
```
