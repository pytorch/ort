import copy
import functools
import inspect
import os
import random
import traceback
import types
import warnings
from distutils.version import LooseVersion
from typing import List

import numpy as np
import torch
from onnx import TensorProto
from torch._C import _from_dlpack
from torch.utils.dlpack import from_dlpack, to_dlpack

from onnxruntime.capi import _pybind_state as C
from onnxruntime.capi.onnxruntime_inference_collection import OrtValue
#from onnxruntime.tools import pytorch_export_contrib_ops

from onnxruntime.training.ortmodule import _onnx_models
#from ._torch_module_pytorch import TorchModulePytorch
#from .torch_cpp_extensions.cpu.aten_op_executor import load_aten_op_executor_cpp_extension

def patch_ortinferencemodule_forward_method(ortinferencemodule):
    # Create forward dynamically, so each ORTInferenceModule instance will have its own copy.
    # This is needed to be able to copy the forward signatures from the original PyTorch models
    # and possibly have different signatures for different instances.
    def _forward(self, *inputs, **kwargs):
        """Forward pass starts here and continues at `_ORTInferenceModuleFunction.forward`
        ONNX model is exported the first time this method is executed.
        Next, we instantiate the ONNX Runtime InferenceSession.
        """
        return ortinferencemodule.forward(*inputs, **kwargs)

    # Bind the forward method.
    ortinferencemodule.forward = _forward.__get__(ortinferencemodule)
    # Copy the forward signature from the PyTorch module
    functools.update_wrapper(ortinferencemodule.forward.__func__, ortinferencemodule._original_module.forward.__func__)
