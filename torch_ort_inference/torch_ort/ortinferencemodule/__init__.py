# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import sys
import warnings
from cmath import e

import torch
from packaging import version

from onnxruntime import set_seed
from onnxruntime.capi import build_and_package_info as ort_info


def _defined_from_envvar(name, default_value, warn=True):
    new_value = os.getenv(name, None)
    if new_value is None:
        return default_value
    try:
        new_value = type(default_value)(new_value)
    except (TypeError, ValueError) as e:
        if warn:
            warnings.warn("Unable to overwrite constant %r due to %r." % (name, e))
        return default_value
    return new_value


################################################################################
# All global constant goes here, before ORTInferenceModule is imported ##################
# NOTE: To *change* values in runtime, import onnxruntime.training.ortmodule and
# assign them new values. Importing them directly do not propagate changes.
################################################################################
ONNX_OPSET_VERSION = 14
MINIMUM_RUNTIME_PYTORCH_VERSION_STR = "1.8.1"

# Verify minimum PyTorch version is installed before proceding to ONNX Runtime initialization
try:
    import torch

    runtime_pytorch_version = version.parse(torch.__version__.split("+")[0])
    minimum_runtime_pytorch_version = version.parse(MINIMUM_RUNTIME_PYTORCH_VERSION_STR)
    if runtime_pytorch_version < minimum_runtime_pytorch_version:
        raise RuntimeError(
            "ONNX Runtime ORTInferenceModule frontend requires PyTorch version greater"
            f" or equal to {MINIMUM_RUNTIME_PYTORCH_VERSION_STR},"
            f" but version {torch.__version__} was found instead."
        )
except ImportError as e:
    raise RuntimeError(
        f"PyTorch {MINIMUM_RUNTIME_PYTORCH_VERSION_STR} must be "
        "installed in order to run ONNX Runtime ORTInferenceModule frontend!"
    ) from e

# Initalized ORT's random seed with pytorch's initial seed
# in case user has set pytorch seed before importing ORTInferenceModule
set_seed((torch.initial_seed() % sys.maxsize))


# Override torch.manual_seed and torch.cuda.manual_seed
def override_torch_manual_seed(seed):
    set_seed(int(seed % sys.maxsize))
    return torch_manual_seed(seed)


torch_manual_seed = torch.manual_seed
torch.manual_seed = override_torch_manual_seed

from onnxruntime.training.ortmodule.debug_options import DebugOptions, LogLevel  # noqa: E402

from .ortinferencemodule import ORTInferenceModule  # noqa: E402
from .provider_options import ProviderOptions, OpenVINOProviderOptions
