The torch-ort packages uses the PyTorch APIs to accelerate PyTorch models using ONNX Runtime.

## Dependencies

The torch-ort package depends on the onnxruntime-training package, which depends on specific versions of GPU libraries such as NVIDIA CUDA.

The default command `pip install torch-ort` installs the onnxruntime-training version that depends on CUDA 10.2.

If you have a different version of CUDA installed, you can install a different version of onnxruntime-training explicitly:

* CUDA 11.1 `pip install onnxruntime-training -f https://download.onnxruntime.ai/onnxruntime_stable_cu111.html`

## Post-installation step

Once torch-ort is installed, there is a post-installation step:

`python -m torch_ort.configure`

If this step fails, it is likely due to GPU library version mismatch between onnxruntime-training and your installation. You can check the version of onnxruntime-training by running `pip list`. For example:

```
onnxruntime-training 1.9.0+cu111
```

## Releases

* 1.9.0

  Release Notes : https://github.com/pytorch/ort/releases/tag/v1.9.0

* 1.8.1

  Release Notes : https://github.com/pytorch/ort/releases/tag/v1.8.1

