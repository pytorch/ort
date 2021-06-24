# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from enum import Enum
from onnxruntime.capi._pybind_state import PropagateCastOpsStrategy
from torch_ort import ORTModule


class PropagateCastLevel(Enum):
     NOT_USED = -1
     FASTER_KEEP_PRECISION = 1
     AGGRRESSIVE_MIXED_PRECISION  = 2

def set_propagate_cast_ops_optimization(model: ORTModule,
                                        strategy : PropagateCastOpsStrategy=PropagateCastOpsStrategy.FLOOD_FILL,
                                        level: PropagateCastLevel=PropagateCastLevel.NOT_USED) -> None:
    '''Set Cast Op propagation strategy for ONNX graph optimization in an attempt to achieve higher throughput

    Cast Op propagation allows ONNX graph to be optimized by replacing float32 nodes by their 16-bit counterpart
    without losing precision during computation. To enable cast propagation, user must select `strategy` and a `level`.
    Each combination of strategy and level have predefined lists of allowed nodes that are safe to move float-cast
    operations from inputs to outputs and float16-cast operations from outputs to inputs.

    Args:
        model (ORTModule): ORTModule instance to apply the cast propagation configuration
        strategy (PropagateCastOpsStrategy, default is FLOOD_FILL): specify the choice of cast propagation optimization.
            It must be one of NONE, INSERT_AND_REDUCE, FLOOD_FILL or REMOVE_INPUT_OUTPUT_UP_DOWN_CASTS.
            NONE strategy doesn't perform any cast propagation transformation on the graph, although other optimizations
            locally change cast operations. For example, in order to fuse Transpose and MatMul nodes,
            the TransposeMatMulFunsion optimization could interchange Transpose and Cast if the Cast node exists
            between Transpose and MatMul.
            INSERT_AND_REDUCE strategy inserts and reduces cast operations around nodes with a predefined list of
            allowed nodes, even if that results in changing nodes outside the expanded float16 region.            
            FLOOD_FILL strategy expands float16 regions using the a predefined allowed list of nodes without modifying
            nodes outside the expanded float16 region.
            REMOVE_INPUT_OUTPUT_UP_DOWN_CASTS strategy removes float Cast on inputs and float16 Casts on outputs for
            nodes of any operations to increase throughput. For example, if both inputs of a node with Add operation,
            happen to be outputs of float Cast operations and the output happen to be input to a float16 Cast operation,
            requiring the Add operation to be performed in float instead of float16, then it is possible to remove casts
            on inputs and output to perform the Add operation in float16 to increase throughput.
            The pattern of up/down casts on inputs/outputs could be due to other cast propagation optimizations.
        level (PropagateCastLevel, default is NOT_USED): NOT_USED does not optimize the graph.
            FASTER_KEEP_PRECISION and AGGRRESSIVE_MIXED_PRECISION use predetermined list of nodes considered safe to
            move before/after cast operation. While FASTER_KEEP_PRECISION guarantees precision is not affected,
            AGGRRESSIVE_MIXED_PRECISION usually affects final precision.

    Raises:
        TypeError: if :attr:`model` is not a :class:`ORTModule` object
        TypeError: if :attr:`strategy` is not a :class:`PropagateCastOpsStrategy` object
        TypeError: if :attr:`level` is not a :class:`PropagateCastLevel` object

    '''

    if not isinstance(model, ORTModule):
        raise TypeError(f'`model` must be a `ORTModule` object, but `{type(model)}` object was specified.')

    if not isinstance(strategy, PropagateCastOpsStrategy):
        raise TypeError(f'`strategy` must be a `PropagateCastOpsStrategy` object, but `{type(model)}` object was specified.')

    if not isinstance(level, PropagateCastLevel):
        raise TypeError(f'`level` must be a `PropagateCastLevel` object.')

    # Set flags for both eval and training mode
    for mode in [True, False]:
        model._execution_manager(is_training=mode)._propagate_cast_ops_strategy = strategy
        model._execution_manager(is_training=mode)._propagate_cast_ops_level = level
