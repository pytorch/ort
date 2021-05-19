from onnxruntime.capi._pybind_state import PropagateCastOpsStrategy
from torch_ort import ORTModule


def set_propagate_cast_ops_optimization(model: ORTModule,
                                        strategy : PropagateCastOpsStrategy=PropagateCastOpsStrategy.NONE,
                                        level: int=-1) -> None:
    '''Set Cast Op propagation strategy for ONNX graph optimization in an attempt to achieve higher throughput

    Cast Op propagation allows ONNX graph to be optimized by replacing 32-bit operations by their 16-bit counterpart
    without losing precision during computation. To enable cast propagation, user must select `strategy` and a `level`.

    Args:
        model (ORTModule): ORTModule instance to apply the cast propagation configuration
        strategy (PropagateCastOpsStrategy, default is NONE): specify the choice of the cast propagation optimization.
            It must be one of NONE, INSERT_AND_REDUCE, FLOOD_FILL or REMOVE_INPUT_OUTPUT_UP_DOWN_CASTS.
            NONE strategy doesn't perform any cast propagation transformation on the graph, although other optimizations
            locally change cast operations. For example, in order to fuse Transpose and MatMul nodes,
            the TransposeMatMulFunsion optimization could interchange Transpose and Cast if the Cast node exists
            between Transpose and MatMul.
            INSERT_AND_REDUCE strategy inserts and reduces cast operations around the nodes with allowed nodes.
            FLOOD_FILL strategy expands float16 regions using the allowed nodes, and unlike INSERT_AND_REDUCE,
            does not touch opcodes outside expanded float16 region.
            REMOVE_INPUT_OUTPUT_UP_DOWN_CASTS strategy removes float Cast on inputs and float16 Casts on outputs for
            nodes of any operations to increase throughput. For example, if both inputs of a node with Add operation,
            happen to be outputs of float Cast operations and the output happen to be input to a float16 Cast operation,
            requiring the Add operation to be performed in float instead of float16, then it is possible to remove casts
            on inputs and output to perform the Add operation in float16 to increase throughput.
            This pattern of up/down casts on inputs/outputs could be due to other cast propagation optimizations.
            If this strategy flag is not set then the transformation will not be performed.
        level (int, default is -1): Level -1 does not optimize the graph by moving Cast operations.
            Level 1 and 2 use predetermined list of nodes considered safe to move before/after cast operation. While
            level 1 guarantees precision is not affected, level 2 usually affects final precision.

    Raises:
        TypeError: if :attr:`model` is not a :class:`ORTModule` object
        TypeError: if :attr:`strategy` is not a :class:`PropagateCastOpsStrategy` object
        TypeError: if :attr:`level` is not a :class:`int` object

    '''

    if not isinstance(model, ORTModule):
        raise TypeError(f'`model` must be a `ORTModule` object, but `{type(model)}` object was specified.')

    if not isinstance(strategy, PropagateCastOpsStrategy):
        raise TypeError(f'`strategy` must be a `PropagateCastOpsStrategy` object, but `{type(model)}` object was specified.')

    if not isinstance(level, int) or level < -1 or level > 2:
        raise TypeError(f'`level` must be an integer between (-1, 2).')

    # Set flags for both eval and training mode
    for mode in [True, False]:
        model._execution_manager(is_training=mode)._propagate_cast_ops_strategy = strategy
        model._execution_manager(is_training=mode)._propagate_cast_ops_level = level
