# Install torch-ort-infer

You can install and run torch-ort-infer in your local environment.

## Prerequisites

- Ubuntu 18.04, 20.04

- Python* 3.7, 3.8 or 3.9

## Run in a Python environment

### Default dependencies

By default, torch-ort-inference depends on PyTorch 1.12 and ONNX Runtime OpenVINO EP 1.12.

Install torch-ort-infer with OpenVINO dependencies

- `pip install torch-ort-infer[openvino]`

## Test your installation

Once you have created your environment, using Python, execute the following steps to validate that your installation is correct.

1. Download a inference script

    `wget https://raw.githubusercontent.com/pytorch/ort/main/torch_ort_inference/demos/bert_for_sequence_classification.py`

2. Install extra dependencies

    `pip install wget pandas transformers`

3. Run the inference script

    `python ./ort/torch_ort_inference/tests/bert_for_sequence_classification.py`

4. Expected warnings

   There are some warnings that are expected.
