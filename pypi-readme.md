The torch-ort packages uses the PyTorch APIs to accelerate PyTorch models using ONNX Runtime.

## Dependencies

The torch-ort package depends on the onnxruntime-training package, which depends on specific versions of GPU libraries such as NVIDIA CUDA.

The default command`pip install torch-ort` will install the onnxruntime-training version that depends on CUDA 10.2.

If you have a different version of CUDA installed, you can install a different version of onnxruntime-training explicitly:

* CUDA 11.1 `pip install onnxruntime-training -f https://download.onnxruntime.ai/onnxruntime_stable_cu111.html` 


## Releases

* 1.9.0

  Release Notes : https://github.com/pytorch/ort/releases/tag/v1.9.0

* 1.8.1

  Release Notes : https://github.com/pytorch/ort/releases/tag/v1.8.1

