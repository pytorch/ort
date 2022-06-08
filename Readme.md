This repository contains below libraries. They can be installed independent of each other.
* [ort_moe](#mixture-of-experts): Mixture of Experts implementation in PyTorch
* [torch_ort](#accelerate-pytorch-models-with-onnx-runtime): ONNX Runtime package that accelerates PyTorch models
* [torch_ort_inference](#accelerate-inference-for-pytorch-models-with-onnx-runtime-(Preview)): ONNX Runtime package that accelerates inference for PyTorch models


# Mixture Of Experts

Mixture of Experts layer implementation is available in the [ort_moe](ort_moe) folder. 
- [ort_moe/docs/moe.md](ort_moe/docs/moe.md) provides brief overview of the implementation.
- A simple MoE tutorial is provided [here](ort_moe/docs/tutorials/moe_tutorial.py).
- Note: ONNX Runtime (following pre-requisites) is not required to run the MoE layer. It is intergrated in stand-alone Pytorch.

## Build MoE
```
cd ort_moe
pip install build # Install PyPA build
python -m build
```

# Accelerate PyTorch models with ONNX Runtime

ONNX Runtime for PyTorch accelerates PyTorch model training using ONNX Runtime.

It is available via the torch-ort python package.

This repository contains the source code for the package, as well as instructions for running the package.

## Pre-requisites

You need a machine with at least one NVIDIA or AMD GPU to run ONNX Runtime for PyTorch.

You can install and run torch-ort in your local environment, or with [Docker](torch_ort/docker/README.md).

## Install in a local Python environment

### Default dependencies

By default, torch-ort depends on PyTorch 1.9.0, ONNX Runtime 1.9.0 and CUDA 10.2.

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

# Accelerate inference for PyTorch models with ONNX Runtime (Preview)

ONNX Runtime for PyTorch accelerates PyTorch model inference using ONNX Runtime.

It is available via the torch-ort-inference python package. This preview package enables OpenVINO™ Execution Provider for ONNX Runtime by default for accelerating inference on various Intel CPUs and integrated GPUs.

This repository contains the source code for the package, as well as instructions for running the package.

## Install in a local Python environment

By default, torch-ort-inference depends on PyTorch 1.12 and ONNX Runtime OpenVINO EP 1.11.

Install torch-ort-inference

- `pip install torch-ort-inference`

## Verify your installation

Once you have created your environment, using Python, execute the following steps to validate that your installation is correct.

1. Download a inference script

   - `wget https://raw.githubusercontent.com/pytorch/ort/main/torch_ort_inference/tests/bert_for_sequence_classification.py`

2. Install extra dependencies

    - `pip install wget pandas sklearn transformers`

3. Run the inference script

    - `python ./ort/torch_ort_inference/tests/bert_for_sequence_classification.py`

## Add ONNX Runtime for PyTorch to your PyTorch inference script

```python
from torch_ort import ORTInferenceModule
model = ORTInferenceModule(model)

# PyTorch inference script follows
```

### Provider Options

Users can configure different options  for a given Execution Provider to run inference. As an example, OpenVINO™ Execution Provider options can be configured as shown below:

```python
from torch_ort import ORTInferenceModule, OpenVINOProviderOptions
provider_options = OpenVINOProviderOptions(backend = "GPU", precision = "FP16")
model = ORTInferenceModule(model, provider_options = provider_options)

# PyTorch inference script follows
```

If no provider options are specified by user, OpenVINO™ Execution Provider is enabled with following options by default.

```python
backend = "CPU"
precision = "FP32"
```

## License

This project has an MIT license, as found in the [LICENSE](LICENSE) file.
