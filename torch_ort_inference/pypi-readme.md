[OpenVINO™ Integration with Torch-ORT](https://github.com/pytorch/ort#-inference) accelerates PyTorch models using OpenVINO™ Execution Provider for ONNX Runtime. This product is designed for PyTorch developers who want to get started with OpenVINO™ in their inferencing applications. It delivers OpenVINO™ inline optimizations that enhance inferencing performance with minimal code modifications.

OpenVINO™ Integration with Torch-ORT accelerates inference across many AI models on a variety of Intel® hardware such as:

* Intel® CPUs
* Intel® integrated GPUs
* Intel® Movidius™ Vision Processing Units - referred to as VPU.

## Installation

### Requirements
* Ubuntu 18.04, 20.04
* Python 3.7, 3.8 or 3.9

#### This package supports:
* Intel® CPUs
* Intel® integrated GPUs
* Intel® Movidius™ Vision Processing Units (VPUs).

The torch-ort-infer package has dependency on the onnxruntime-openvino package that will be installed by default to run inference workloads. This onnxruntime-openvino package comes with pre-built libraries of OpenVINO™ version 2022.2.0 eliminating the need to install OpenVINO™ separately. The OpenVINO™ libraries are prebuilt with CXX11_ABI flag set to 0.

For more details, please refer to [OpenVINO™ Execution Provider for ONNX Runtime.](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html)

## Post-installation step

Once torch-ort-infer is installed, there is a post-installation step:

`python -m torch_ort.configure`

## Usage

By default, Intel® CPU is used to run inference. However, you can change the default option to either Intel® integrated GPU or Intel® VPU for AI inferencing. Invoke the [provider options](https://github.com/pytorch/ort/blob/main/torch_ort_inference/docs/usage.md#essential-apis) to change the hardware on which inferencing is done.

For more API calls and environment variables, see [Usage](https://github.com/pytorch/ort/blob/main/torch_ort_inference/docs/usage.md).

## Samples
For quick start, explore the [samples](https://github.com/pytorch/ort/tree/main/torch_ort_inference/demos) for few HuggingFace and TorchVision models.

## License
OpenVINO™ Integration with Torch-ORT is licensed under [MIT](https://github.com/pytorch/ort/blob/main/LICENSE). By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.

## Support
Please submit your questions, feature requests and bug reports via [GitHub Issues](https://github.com/pytorch/ort/issues).

## How to Contribute
We welcome community contributions to OpenVINO™ Integration with Torch-ORT. If you have an idea for improvement:

* Share your proposal via [GitHub Issues](https://github.com/pytorch/ort/issues).
* Submit a [Pull Request](https://github.com/pytorch/ort/pulls).



