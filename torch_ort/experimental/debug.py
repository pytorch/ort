import os
from torch_ort import ORTModule
from onnxruntime.training.ortmodule._logger import LogLevel


def save_intermediate_onnx_graphs(model: ORTModule, prefix: str) -> bool:
    '''Saves all intermediate ONNX graphs produced by `model` at the specified `prefix`

    When `prefix` is either None or an empty string, ONNX graphs saving is disabled, otherwise
    multiple ONNX graphs will be saved as specified by `prefix` with an ending ".onnx" extension.

    Args:
        model (ORTModule): ORTModule instance
        prefix(str): Full path plus a file prefix to be prepended to all ONNX graphs.
                     Prefix must have a directory and file prefix components (e.g. /foo/bar_).
                     The directory must exist and be writable by the user.

    Returns:
        True if ONNX graph save is enabled or False otherwise

    Raises:
        NotADirectoryError: if :attr:`prefix`does not contain a directory component
        NameError: if :attr:`prefix`does not contain a file prefix component
        TypeError: if :attr:`model` is not a :class:`ORTModule` object

    '''

    if not isinstance(model, ORTModule):
        raise TypeError(f'`model` must be a `ORTModule` object, but `{type(model)}` object was specified.')

    assert prefix is None or isinstance(prefix, str), '`prefix` must be a non-empty string or `None`.'

    dir_name = os.path.dirname(prefix)
    prefix_name = os.path.basename(prefix)

    if not os.path.exists(dir_name):
        raise NotADirectoryError(f'{dir_name} is not a valid directory to save ONNX graphs.')

    if not prefix_name or prefix_name.isspace():
        raise NameError(f'{prefix_name} is not a valid prefix name for the ONNX graph files.')

    enable_save = dir_name and prefix_name
    for mode in [True, False]:
        model._execution_manager(is_training=mode)._save_onnx = enable_save
        model._execution_manager(is_training=mode)._save_onnx_prefix = prefix

    return enable_save


def set_log_level(model: ORTModule, level: LogLevel) -> None:
    '''Set verbosity level for `model` at the specified `level`

    Args:
        model (ORTModule): ORTModule instance
        level(LogLevel): Log level which must be one of ().

    Returns:
        True if ONNX graph save is enabled or False otherwise

    Raises:
        TypeError: if :attr:`model` is not a :class:`ORTModule` object
        TypeError: if :attr:`level` is not a :class:`LogLevel` object

    '''

    if not isinstance(model, ORTModule):
        raise TypeError(f'`model` must be a `ORTModule` object, but `{type(model)}` object was specified.')

    if not isinstance(level, LogLevel):
        raise TypeError(f'`level` must be a `LogLevel` object, but `{type(level)}` object was specified.')

    for mode in [True, False]:
        model._execution_manager(mode)._loglevel = level
