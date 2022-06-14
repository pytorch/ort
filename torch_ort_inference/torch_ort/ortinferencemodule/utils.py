import functools

def get_device_from_module(module):
    """Returns the first device found in the `module`'s parameters or None
    Args:
        module (torch.nn.Module): PyTorch model to extract device from
    Raises:
        Exception: When more than one device is found at `module`
    """
    device = None
    try:
        device = next(module.parameters()).device
        for param in module.parameters():
            if param.device != device:
                raise e(
                    RuntimeError("ORTInferenceModule supports a single device per model")
                )
    except Exception as e:
            raise e
        # Model doesn't have a device set to any of the model parameters
    return device

def patch_ortinferencemodule_forward_method(ortinferencemodule):
    # Create forward dynamically, so each ORTInferenceModule instance will have its own copy.
    # This is needed to be able to copy the forward signatures from the original PyTorch models
    # and possibly have different signatures for different instances.
    def _forward(self, *inputs, **kwargs):
        """Forward pass starts here and continues at `_ORTInferenceModuleFunction.forward`
        ONNX model is exported the first time this method is executed.
        Next, we instantiate the ONNX Runtime InferenceSession
        Finally, we run the ONNX Runtime InferenceSession.
        """
        return ortinferencemodule._forward_call(*inputs, **kwargs)

    # Bind the forward method.
    ortinferencemodule.forward = _forward.__get__(ortinferencemodule)
    # Copy the forward signature from the PyTorch module
    functools.update_wrapper(ortinferencemodule.forward.__func__, ortinferencemodule._original_module.forward.__func__)


def set_dynamic_axes(module):

    """Returns true if dynamic_axes parameter needs to be set for
    onnx export
    Args:
        module (torch.nn.Module): Pytorch model
    """
    try:
        avg_pool_module = module._original_module.get_submodule("avgpool")
        output_size = list(avg_pool_module.output_size)
        if output_size != [1] * len(output_size):
            return False
    except Exception:
        return True
    return True
