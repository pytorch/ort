<div align="center">

<p align="center"><img width="50%" src="images/ONNX_Runtime_logo_dark.png" /></p>

**A library for developing and deploying PyTorch models using the ONNX Runtime**.

---

[Key Features](#key-features) •
[Installation](#Installation) •
[Training](#Training) •
[Inference](#Inference) •
[Docs](https://www.onnxruntime.ai/) •
[License](https://github.com/pytorch/ort/blob/main/LICENSE)

[![torch-ort](https://img.shields.io/pypi/v/torch-ort)](https://pypi.org/project/torch-ort/)
[![python](https://img.shields.io/badge/python-3.7%2B-blue)]()
[![pytorch](https://img.shields.io/badge/pytorch-1.12.1%2B-blue)]()
[![API Checks](https://github.com/pytorch/ort/actions/workflows/api-ci.yml/badge.svg)](https://github.com/pytorch/ort/actions/workflows/api-ci.yml)
[![Docs](https://github.com/pytorch/ort/actions/workflows/doc-automation.yml/badge.svg)](https://github.com/pytorch/ort/actions/workflows/doc-automation.yml)

</div>

---

# Introduction

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

**Key Features:**

- Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
- Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
- Inference PyTorch directly from PyTorch framework while leveraging ONNX Runtime execution provider optimizations

# 🚀 Installation

## Inference

### Prerequisites

- Ubuntu 18.04, 20.04
- Python* 3.7, 3.8 or 3.9

### Install in a local Python environment
- `pip install torch-ort-infer[openvino]`
- Run post installation configuration script `python -m torch_ort.configure`

### Verify your installation

1. Clone this repo

    - `git clone git@github.com:pytorch/ort.git`

2. Install extra dependencies

    - `pip install wget pandas sklearn transformers`

3. Run the training script

    - `python ./torch_ort_inference/tests/bert_for_sequence_classification.py`

## Training

### Pre-requisites

You need a machine with at least one NVIDIA or AMD GPU to run ONNX Runtime for PyTorch.

You can install and run torch-ort in your local environment, or with [Docker](torch_ort/docker/README.md).

### Install in a local Python environment


By default, torch-ort depends on PyTorch 1.9.0, ONNX Runtime 1.9.0 and CUDA 10.2.

1. Install CUDA 10.2

2. Install CuDNN 7.6

3. Install torch-ort

    - `pip install torch-ort`

4. Run post-installation script for ORTModule

    - `python -m torch_ort.configure`

Get install instructions for other combinations in the `Get Started Easily` section at <https://www.onnxruntime.ai/> under the `Optimize Training` tab.

### Verify your installation

1. Clone this repo

    - `git clone git@github.com:pytorch/ort.git`

2. Install extra dependencies

    - `pip install wget pandas sklearn transformers`

3. Run the training script

    - `python ./ort/tests/bert_for_sequence_classification.py`


## Inference

### Prerequisites

- Ubuntu 18.04, 20.04
- Python* 3.7, 3.8 or 3.9

### Install in a local Python environment
- `pip install torch-ort-infer[openvino]`
- Run post installation configuration script `python -m torch_ort.configure`

### Verify your installation

1. Clone this repo

    - `git clone git@github.com:pytorch/ort.git`

2. Install extra dependencies

    - `pip install wget pandas sklearn transformers`

3. Run the training script

    - `python ./torch_ort_inference/tests/bert_for_sequence_classification.py`

# 📈 Training

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

## Samples

To see torch-ort in action, see https://github.com/microsoft/onnxruntime-training-examples, which shows you how to train the most popular HuggingFace models.

# 🎯 Inference

<div align="center">

<p align="center"><img width="30%" src="images/ONNX_Runtime_logo_dark.png" /></p>
<p align="center">➕</p>
<p align="center"><img width="30%" src="images/openvino-logo-purple-black.png" /></p>

</div>


ONNX Runtime for PyTorch is now extended to support PyTorch model inference using ONNX Runtime.

It is available via the torch-ort-infer python package. This preview package enables OpenVINO™ Execution Provider for ONNX Runtime by default for accelerating inference on various Intel® CPUs, Intel® integrated GPUs, and Intel® Movidius™ Vision Processing Units - referred to as VPU.

## Supported Execution Providers

|Execution Providers|
|---|
|OpenVINO  [![openvino](https://img.shields.io/badge/openvino-2022.1-purple)]() |

## Provider Options

Users can configure different options  for a given Execution Provider to run inference. As an example, OpenVINO™ Execution Provider options can be configured as shown below:

```python
from torch_ort import ORTInferenceModule, OpenVINOProviderOptions
provider_options = OpenVINOProviderOptions(backend = "GPU", precision = "FP16")
model = ORTInferenceModule(model, provider_options = provider_options)

# PyTorch inference script follows
```
### List of Provider Options

Supported backend-precision combinations:
| Backend | Precision |
| --------| --------- |  
|   CPU   |    FP32   |
|   GPU   |    FP32   |
|   GPU   |    FP16   |
|  MYRIAD |    FP16   |

If no provider options are specified by user, OpenVINO™ Execution Provider is enabled with following options by default:

```python
backend = "CPU"
precision = "FP32"
```

For more details on APIs, see [usage.md](/torch_ort_inference/docs/usage.md).

## Code Sample

Below is an example of how you can leverage OpenVINO™ integration with Torch-ORT in a simple NLP usecase. 
A pretrained [BERT model](https://huggingface.co/textattack/bert-base-uncased-CoLA) fine-tuned on the CoLA dataset from HuggingFace model hub is used to predict grammar correctness on a given input text. 


```python 
from transformers 
import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from torch_ort import ORTInferenceModule
tokenizer = AutoTokenizer.from_pretrained(
            "textattack/bert-base-uncased-CoLA")
model = AutoModelForSequenceClassification.from_pretrained(
        "textattack/bert-base-uncased-CoLA")
# Wrap model in ORTInferenceModule to prepare the model for inference using OpenVINO Execution Provider on CPU
model = ORTInferenceModule(model)
text = "Replace me any text by you'd like ."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
# Post processing
logits = output.logits
logits = logits.detach().cpu().numpy()
# predictions
pred = np.argmax(logits, axis=1).flatten()
print("Grammar correctness label (0=unacceptable, 1=acceptable)")
print(pred)
```

## Samples

To see OpenVINO™ integration with Torch-ORT in action, see [demos](/torch_ort_inference/demos), which shows you how to run inference on some of the most popular Deep Learning models.


# 🤝 Contributing

Please refer to our [contributing guide](CONTRIBUTING.md) for more information on how to contribute!

## License

This project has an MIT license, as found in the [LICENSE](LICENSE) file.
