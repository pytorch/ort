# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
from torch_ort import ORTModule
from onnxruntime.training.ortmodule._logger import LogLevel


def save_intermediate_onnx_graphs(model: ORTModule, prefix: str, enable: bool=True) -> None:
    '''Saves all intermediate ONNX graphs produced by `model` at the specified `prefix`

    When `enable` is `False`, ONNX graphs saving is disabled, otherwise
    multiple ONNX graphs will be saved as specified by `prefix` with an ending ".onnx" extension.

    Args:
        model (ORTModule): ORTModule instance
        prefix(str): Full path plus a file prefix to be prepended to all ONNX graphs.
                     Prefix must have a directory and file prefix components (e.g. /foo/bar_).
                     The directory must exist and be writable by the user.
        enable (bool, default is True): Toggle to enable or disable ONNX graph saving

    Raises:
        NotADirectoryError: if :attr:`prefix`does not contain a directory component
        NameError: if :attr:`prefix`does not contain a file prefix component
        TypeError: if :attr:`model` is not a :class:`ORTModule` object

    '''

    if not isinstance(model, ORTModule):
        raise TypeError(f'`model` must be a `ORTModule` object, but `{type(model)}` object was specified.')

    if not isinstance(prefix, str):
        raise TypeError('`prefix` must be a non-empty string.')

    if not isinstance(enable, bool):
        raise TypeError('`enable` must be a boolean.')

    dir_name = os.path.dirname(prefix)
    prefix_name = os.path.basename(prefix)

    if not os.path.exists(dir_name):
        raise NotADirectoryError(f'{dir_name} is not a valid directory to save ONNX graphs.')

    if not prefix_name or prefix_name.isspace():
        raise NameError(f'{prefix_name} is not a valid prefix name for the ONNX graph files.')

    # Set flags for both eval and training mode
    for mode in [True, False]:
        model._execution_manager(is_training=mode)._save_onnx = enable
        model._execution_manager(is_training=mode)._save_onnx_prefix = prefix


def set_log_level(model: ORTModule, level: LogLevel=LogLevel.WARNING) -> None:
    '''Set verbosity level for `model` at the specified `level`

    Args:
        model (ORTModule): ORTModule instance
        level(LogLevel, default is WARNING): Log level which must be one of VERBOSE, INFO, WARNING, ERROR, FATAL.

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
