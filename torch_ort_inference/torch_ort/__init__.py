# -------------------------------------------------------------------------
# Copyright (C) 2022 Intel Corporation
# Licensed under the MIT License
# --------------------------------------------------------------------------

from onnxruntime.training.ortmodule import DebugOptions, LogLevel
from .ortinferencemodule import ORTInferenceModule
from .ortinferencemodule.provider_options import OpenVINOProviderOptions

# Add a __version__ attribute that users can use to check the version
try:
    from ._version import __version__
except ModuleNotFoundError:
    # ._version.py file not found. Not setting any __version__ attribute
    pass
