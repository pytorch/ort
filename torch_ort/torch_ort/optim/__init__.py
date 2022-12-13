# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""
:mod:`torch_ort.optim` is a package that expose various optimization algorithms to the ORTModule.
"""

from onnxruntime.training.optim import FusedAdam
from onnxruntime.training.optim.fp16_optimizer import FP16_Optimizer
