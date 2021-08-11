# Accelerate PyTorch models with ONNX Runtime

ONNX Runtime for PyTorch accelerates PyTorch model training using ONNX Runtime.

It is available via the torch-ort python package.

This repository contains the source code for the package, as well as instructions for running the package.

## Pre-requisites

You need a machine with at least one NVIDIA or AMD GPU to run ONNX Runtime for PyTorch.

You can install and run torch-ort in your local environment, or with [Docker](docker/README.md).

## Install in a local Python environment

### Default dependencies

By default, torch-ort depends on PyTorch 1.9.0, ONNX Runtime 1.8.1 and CUDA 10.2.

1. Install CUDA 10.2

2. Install CuDNN 7.6

3. Install torch-ort

    - `pip install torch-ort`

4. Run post-installation script for ORTModule

    - `python -m torch_ort.configure`

Get install instructions for other combinations in the `Get Started Easily` section at <https://www.onnxruntime.ai/> under the `Optimize Training` tab.

## Verify your installation

1. Clone this repo

    - `git clone git@github.com:pytorch/ort.git`

2. Install extra dependencies

    - `pip install wget pandas sklearn transformers`

3. Run the training script

    - `python ./ort/tests/bert_for_sequence_classification.py`

## Add ONNX Runtime for PyTorch to your PyTorch training script

```python
from torch_ort import ORTModule
model = ORTModule(model)

# PyTorch training script follows
```

## Samples

To see torch-ort in action, see https://github.com/microsoft/onnxruntime-training-examples, which shows you how to train the most popular HuggingFace models.

## License

This project has an MIT license, as found in the [LICENSE](LICENSE) file.
