# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from onnxruntime.training.ortmodule import ORTModule, DebugOptions, LogLevel
from onnxruntime import set_seed

# Add a __version__ attribute that users can use to check the version
try:
    from ._version import __version__
except ModuleNotFoundError:
    # ._version.py file not found. Not setting any __version__ attribute
    pass
