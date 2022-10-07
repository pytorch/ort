# OpenVINO™ integration with Torch-ORT Dockerfiles for Ubuntu* 18.04 and Ubuntu* 20.04


We provide Dockerfiles for Ubuntu* 18.04 and Ubuntu* 20.04 which can be used to build runtime Docker* images for OpenVINO™ integration with Torch-ORT on CPU, iGPU and VPU.
They contain all required runtime python packages, and shared libraries to support execution of a Torch-ORT Python app with the OpenVINO™ backend. By default, it hosts an ResNet Image Classification and BERT Sequence Classification sample that demonstrate the performance benefits of using OpenVINO™ integration with Torch-ORT.

The following ARG is available to configure the docker build:

TORCHORT_BRANCH: OpenVINO™ integration with Torch-ORT branch to be used. Defaults to "main"


### Build the docker image

	cd torch_ort_inference\docker\
    docker build -f Dockerfile.ort-infer-stable-openvino-<distro_name> -t <image_name> .


### For Ubuntu Host System

- To run on **CPU** backend:

		docker run -it --rm <image_name>

- To run on **iGPU** backend: 

		docker run -it --rm \
		--device-cgroup-rule='c 189:* rmw' \
		--device /dev/dri:/dev/dri \
		--group-add=$(stat -c "%g" /dev/dri/render*) \
		<image_name> 

- To run on **MYRIAD** backend:

		docker run -it --rm \
		--device-cgroup-rule='c 189:* rmw' \
		-v /dev/bus/usb:/dev/bus/usb \
		--group-add=$(stat -c "%g" /dev/dri/render*) \
		<image_name>

- Run image with runtime target /bin/bash for container shell with CPU, iGPU, and MYRIAD device access:

		docker run -itu root:root --rm \
		--device-cgroup-rule='c 189:* rmw' \
		--device /dev/dri:/dev/dri \
		--mount type=bind,source=/var/tmp,destination=/var/tmp \
		-v /dev/bus/usb:/dev/bus/usb \
		<image_name> /bin/bash

### For Windows Host System

- To run on **CPU** backend:

		docker run -it --rm <image_name>

- To run inference on **iGPU** backend, the host system is required to be Windows* 10 21H2 or Windows* 11 with [WSL-2](https://docs.microsoft.com/en-us/windows/wsl/install) installed as a pre-requisite and must have [Intel iGPU drivers](https://www.intel.com/content/www/us/en/download/19344/intel-graphics-windows-dch-drivers.html) with version 30.0.100.9684 or above installed :

		docker run -it --rm \
		--device /dev/dxg \
		--volume /usr/lib/wsl:/usr/lib/wsl \
		<image_name>

### For MacOS Host System

- To run on **CPU** backend (Supports CPU only):

		docker run -it --rm <image_name>

### Inside the Docker Container

While running the docker container, users will get a ready-made environment with all the pre-installed dependencies and software artifacts in place.
The users will witness below directory structure listing all the necessary torch_ort_inference software components -

	torch_ort_inference/
    ├── build.py
    ├── demos/
    │   ├── bert_for_sequence_classification.py
    │   ├── bert.md
    │   ├── plane.jpg
    │   ├── resnet_image_classification.py
    │   └── resnet.md
    ├── docker/
    ├── docs/
    ├── pypi-readme.md
    ├── pyproject.toml
    ├── requirements-dev.txt
    ├── requirements.txt
    ├── setup.py
    ├── torch_ort/
    └── version.txt

Users can checkout the demo samples in a running docker container by using `cd demos/` and the detailed receipe for running inference samples can be found in the above listed markdown files viz. [bert.md](ort/torch_ort_inference/demos/bert.md) and [resnet.md](ort/torch_ort_inference/demos/resnet.md) respectively.

For more information on API usage and inference options, refer [usage.md](ort/torch_ort_inference/docs/usage.md)

---
\* Other names and brands may be claimed as the property of others.