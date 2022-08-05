# Bert for Sequence Classification

This demo shows how to use Intel® OpenVINO™ integration with Torch-ORT to check grammar in text with ONNX Runtime OpenVINO Execution Provider.

We use a sequence classification model [textattack/bert-base-uncased-CoLA](https://huggingface.co/textattack/bert-base-uncased-CoLA) from HuggingFace models. This model is trained on the BERT architecture to check grammar.

## Model Metadata
| Domain | Application | Industry  | Framework | Input Data Format |
| ------------- | --------  | -------- | --------- | -------------- | 
| NLP | Sequence Classification | General | PyTorch | Text |

## Prerequisites

- Ubuntu 18.04, 20.04
- Python* 3.7, 3.8 or 3.9

## Install in a local Python environment

1. Upgrade pip

    - `pip install --upgrade pip`
<br/><br/>

2. Install torch-ort-infer with OpenVINO dependencies

    - `pip install torch-ort-infer[openvino]`
<br/><br/>
3. Run post-installation script

    - `python -m torch_ort.configure`

## Verify your installation

Once you have created your environment, execute the following steps to validate that your installation is correct.

1. Clone this repo

    - `git clone https://github.com/pytorch/ort.git`
<br/><br/>
2. Install extra dependencies

    - `pip install wget pandas transformers`
<br/><br/>
3. Run the inference script with default options

    - `python ./ort/torch_ort_inference/demos/bert_for_sequence_classification.py`
<br/><br/>
    **Note**: OpenVINOExecutionProvider is enabled with CPU and FP32 by default.

## Run demo with custom options
```
usage: bert_for_sequence_classification.py [-h] [--pytorch-only] [--input INPUT] [--input-file INPUT_FILE] [--provider PROVIDER] [--backend BACKEND] [--precision PRECISION]

PyTorch BERT Sequence Classification Example

optional arguments:
-h, --help            show this help message and exit
--pytorch-only        disables ONNX Runtime inference
--input "INPUT"         input sentence, put it in quotes
--input-file INPUT_FILE
                        path to input file in .tsv format
--provider PROVIDER   ONNX Runtime Execution Provider
--backend BACKEND     OpenVINO target device (CPU, GPU).
--precision PRECISION
                        OpenVINO target device precision (FP16 or FP32)
```
    
**Note**: Default options and inputs are selected if no arguments are given
    
## Expected output
```
OpenVINOExecutionProvider is enabled with CPU and FP32 by default.
Input not provided! Using default input...

Number of sentences: 2
        Grammar correctness label (0=unacceptable, 1=acceptable)

        'This is a BERT sample.' : 1
        'User input is valid not.' : 0

Average inference time: 25.2306ms
Total Inference time: 50.4613ms
```

<br/><br/>

**Note**: This demo has a warm-up run and then inference time is measured on the subsequent runs. The execution time of first run is in general higher compared to the next runs as it includes inline conversion to ONNX, many one-time graph transformations and optimizations steps.

For more details on APIs, see [usage.md](/torch_ort_inference/docs/usage.md)