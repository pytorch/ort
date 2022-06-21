# -------------------------------------------------------------------------
# Copyright (C) 2022 Intel Corporation
# Licensed under the MIT License
# --------------------------------------------------------------------------

import functools
from collections import abc
from onnxruntime.training.ortmodule import _io

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

def get_user_inputs(onnx_input_names, input_info, inputs, kwargs, device):

    # User inputs
    def _expand_inputs(current_input, non_none_inputs):
        # The exporter handles input lists by expanding them so that each
        # element of the list is its own input.
        # ORTInferenceModule must match this behavior by also expanding the inputs.
        if current_input is None or isinstance(current_input, str):
            # Drop all None and string inputs
            return
        if isinstance(current_input, abc.Sequence):
            # If the input is a sequence (like a list), expand the list so that
            # each element of the list is an input by itself
            for inp in current_input:
                _expand_inputs(inp, non_none_inputs)
        elif isinstance(current_input, abc.Mapping):
            # If the input is a mapping (like a dict), expand the dict so that
            # each element of the dict is an input by itself
            for _, val in current_input.items():
                _expand_inputs(val, non_none_inputs)
        else:
            # else just collect all the non none inputs within non_none_inputs
            non_none_inputs.append(current_input)

    non_none_inputs = []
    _expand_inputs(inputs, non_none_inputs)
    result = []

    for input_idx, name in enumerate(onnx_input_names):
        inp = None
        if name in kwargs and kwargs[name] is not None:
            # Only use keywords coming from user that are expected by ONNX model
            inp = kwargs[name]

        if inp is None:
            try:
                # Only use positionals coming from user that are expected by ONNX model
                # if input_idx >= len(input_info.names), IndexError will be thrown
                if name != input_info.names[input_idx]:
                    # When ONNX drops unused inputs, get correct index from user input
                    # if name is not in input_info.names, ValueError will be thrown
                    input_idx = input_info.names.index(name)
                inp = non_none_inputs[input_idx]
            except (IndexError, ValueError):
                # ONNX input name is not present in input_info.names.
                pass

        if inp is not None:
            if _io._PrimitiveType.is_primitive_type(inp):
                inp = _io._PrimitiveType.get_tensor(inp,device)
            result.append(inp)
        else:
            raise (
                RuntimeError(f"Input is present in ONNX graph but not provided: {name}.")
            )

    return result

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
    # Torchvision models like alexnet, vgg and densenets have AdaptiveAvgPool2d
    # op in the model.
    # This op can be converted through onnx export in two scenarios.
    # 1.If the output_size of this op is 1 for all dimensions,
    # it's converted to GlobalMaxPool.
    # 2. If the output_size is a factor of the input size,
    # then it's converted to MaxPool.
    # However the second condition can only be checked if the model is static
    # and dynamic_axes paramter of torch.onnx.export is not set to True.
    # Hence, we disable dynamic_axes parameter when we cannot convert this op
    # to GlobalMaxPool so that we can check in onnx export if this op can be
    # converted to MaxPool.

    try:
        avg_pool_module = module._original_module.get_submodule("avgpool")
        output_size = list(avg_pool_module.output_size)
        if output_size != [1] * len(output_size):
            return False
    except Exception:
        return True
    return True
