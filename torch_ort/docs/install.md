# Install torch-ort

## Pre-requisites

You need a machine with at least one NVIDIA or AMD GPU to install torch-ort to run ONNX Runtime for PyTorch.

You can install and run torch-ort in your local environment, or with Docker.

## Run in a Python environment

### Default dependencies

By default, torch-ort depends on PyTorch 1.8.1, ONNX Runtime 1.8 and CUDA 10.2.

1. Install CUDA 10.2

2. Install CuDNN 7.6

3. Install torch-ort and dependencies

    - `pip install ninja`
    - `pip install torch-ort`

### Explicitly install for NVIDIA CUDA 10.2

1. Install CUDA 10.2

2. Install CuDNN 7.6

3. Install torch-ort and dependencies

    - `pip install ninja`
    - `pip install torch==1.8.1`
    - `pip install --pre onnxruntime-training -f https://download.onnxruntime.ai/onnxruntime_nightly_cu102.html`
    - `pip install torch-ort`

### Explicitly install for NVIDIA CUDA 11.1

1. Install CUDA 11.1

2. Install CuDNN 8.0

3. Install torch-ort and dependencies

    - `pip install ninja`
    - `pip install torch==1.8.1`
    - `pip install --pre onnxruntime-training -f https://download.onnxruntime.ai/onnxruntime_nightly_cu111.html`
    - `pip install torch-ort`

### Explicitly install for AMD ROCm 4.2

1. Install ROCm 4.2 base package ([instructions](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html))

2. Install ROCm 4.2 libraries ([instructions](https://rocmdocs.amd.com/en/latest/Installation_Guide/Software-Stack-for-AMD-GPU.html#machine-learning-and-high-performance-computing-software-stack-for-amd-gpu-v4-1))

3. Install ROCm 4.2 RCCL ([instructions](https://github.com/ROCmSoftwarePlatform/rccl/tree/rocm-4.2.0))

4. Install torch-ort and dependencies
    - `pip install ninja`
    - `pip install --pre torch -f https://download.pytorch.org/whl/nightly/rocm4.2/torch_nightly.html`
    - `pip install --pre onnxruntime-training -f https://download.onnxruntime.ai/onnxruntime_nightly_rocm42.html`
    - `pip install torch-ort`

## Run using Docker

The [docker](https://github.com/pytorch/ort/tree/main/docker) directory contains dockerfiles for building the environment for ONNX Runtime for PyTorch.

### NVIDIA GPUs

Ensure that the `nvidia-container-toolkit` is installed.

#### CUDA 10.2

1. Build the docker image

    `docker build -f Dockerfile.ort-cu102-cudnn7-devel-ubuntu18.04 -t ort.cu102 .`

2. Run the docker container using the image you have just built

    `docker run -it --gpus all --name my-experiments ort.cu102:latest /bin/bash`

#### CUDA 11.1

1. Build the docker image

    `docker build -f Dockerfile.ort-cu111-cudnn8-devel-ubuntu18.04 -t ort.cu111 .`

2. Run the docker container using the image you have just built

    `docker run -it --gpus all --name my-experiments ort.cu111:latest /bin/bash`

### AMD GPUs

#### ROCm 4.2

1. Build the docker image

    `docker build -f Dockerfile.ort-rocm4.2-pytorch1.8.1-ubuntu18.04 -t ort.rocm42 .`

2. Run the docker container using the image you have just built

    ```bash
    docker run -it --rm \
      --privileged \
      --device=/dev/kfd \
      --device=/dev/dri \
      --group-add video \
      --cap-add=SYS_PTRACE \
      --security-opt seccomp=unconfined \
      --name my-experiments \
      ort.rocm42:latest /bin/bash
    ```

## Test your installation

Once you have created your environment, either using Python or docker, execute the following steps to validate that your installation is correct.

1. Download a training script

    `wget https://raw.githubusercontent.com/pytorch/ort/main/tests/bert_for_sequence_classification.py`

2. Install extra dependencies

    `pip install wget pandas sklearn transformers`

3. Run the training script

    `python ./ort/tests/bert_for_sequence_classification.py`

4. Expected warnings

There are some warnings that are expected.
