# Run torch-ort with Docker

Run torch_ort.ORTModule within a Docker environment.

The instructions below assume you are entering commands from the `docker` folder in this repository. Docker files available for CUDA and ROCm configurations.

## NVIDIA CUDA 11.8

1. Build the docker image

    `docker build -f Dockerfile.onnxruntime-torch2.0.0-cu118-cudnn8-devel-ubuntu20.04 -t ort.cu118 .`

2. Run the docker container using the image you have just built

    `docker run -it --gpus all --name my-experiments ort.cu118:latest /bin/bash`

## AMD ROCm 4.2 (preview)

1. Build the docker image

    `docker build -f Dockerfile.ort-torch181-onnxruntime-stable-rocm4.2-ubuntu18.04 -t ort.rocm42 .`

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

Follow the same instructions to test the local Python installation, available from the main [README](../Readme.md).
