# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Experimental APIs for ORTModule

DO NOT use these APIs for any production code as they can be removed at any time
"""

from .debug import save_intermediate_onnx_graphs, set_log_level
from .graph_config import set_propagate_cast_ops_optimization,\
                          PropagateCastOpsStrategy,\
                          PropagateCastLevel
from onnxruntime.training.ortmodule._logger import LogLevel
