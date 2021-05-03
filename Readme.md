# Train PyTorch models with ONNX Runtime

PyTorch/ORT is a Python package that uses ONNX Runtime to accelerate PyTorch model training.

## Pre-requisites

You need a machine with at least one NVIDIA or AMD GPU to run PyTorch/ORT.

You can install run PyTorch/ORT in your local environment, or with Docker. If you are using Docker, the following base image is suitable for Nvidia and AMD respectively : `nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04` or `rocm/pytorch:rocm4.1.1_ubuntu18.04_py3.6_pytorch`.

## Install for Nvidia GPUs

1. Install CUDA

2. Install CuDNN

3. Install PyTorch/ORT and dependencies
### Nvidia CUDA version 11.1
- `pip install onnx ninja`
- `pip install --pre torch -f https://download.pytorch.org/whl/nightly/cu111/torch_nightly.html`
- `pip install --pre onnxruntime-training -f https://onnxruntimepackages.z14.web.core.windows.net/onnxruntime_nightly_cu111.html`
- `pip install torch-ort`

### Nvidia CUDA version 10.2
- `pip install onnx ninja`
- `pip install --pre torch -f https://download.pytorch.org/whl/nightly/cu102/torch_nightly.html`
- `pip install --pre onnxruntime-training -f https://onnxruntimepackages.z14.web.core.windows.net/onnxruntime_nightly_cu102.html`
- `pip install torch-ort`

## Install for AMD GPUs

1. Install Rocm 4.1 base package ([instructions](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html))

2. Install Rocm 4.1 libraries ([instructions](https://rocmdocs.amd.com/en/latest/Installation_Guide/Software-Stack-for-AMD-GPU.html#machine-learning-and-high-performance-computing-software-stack-for-amd-gpu-v4-1))

3. Install Rocm 4.1 RCCL ([instructions](https://github.com/ROCmSoftwarePlatform/rccl/tree/rocm-4.1.0))

4. Install PyTorch/ORT and dependencies
### AMD ROCM version 4.1
- `pip install onnx ninja`
- `pip install --pre torch -f https://download.pytorch.org/whl/nightly/rocm4.1/torch_nightly.html`
- `pip install --pre onnxruntime-training -f https://onnxruntimepackages.z14.web.core.windows.net/onnxruntime_nightly_rocm41.html`
- `pip install torch-ort`

 to install release package of onnxruntime-training:
 - `pip install onnxruntime-training`
## Test your installation

1. Clone this repo

- `git clone git@github.com:pytorch/ort.git`

2. Install extra dependencies

- `pip install wget pandas sklearn transformers`

3. Run the training script

- `python ./ort/tests/bert_for_sequence_classification.py`

## Add PyTorch/ORT to your PyTorch training script

```python
import onnxruntime
from torch_ort import ORTModule
model = ORTModule(model)
# PyTorch training script follows
```

## Versioning

### CUDA

The PyTorch/ORT package was built with CUDA 11.1. If you have a different version of CUDA installed, you should install the CUDA 11.1 toolkit.

