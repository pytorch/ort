# Run torch-ort with Docker

Instructions below assume you are entering commands from the `docker` folder in this repository. Docker files available for CUDA and ROCm configurations.


### On NVIDIA CUDA 11.1

1. Build the docker image

    `docker build -f Dockerfile.ort-torch190-onnxruntime181-cu111-cudnn8-devel-ubuntu18.04 -t ort.cu111 .`

2. Run the docker container using the image you have just built

    `docker run -it --gpus all --name my-experiments ort.cu111:latest /bin/bash`

## On AMD ROCm 4.2 (preview)

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