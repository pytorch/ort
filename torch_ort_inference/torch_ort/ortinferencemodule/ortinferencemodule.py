# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import copy
import io
from typing import TypeVar

import onnx
import torch
from torch.onnx import OperatorExportTypes

import onnxruntime
from onnxruntime.capi import _pybind_state as C
from onnxruntime.training import ortmodule
from onnxruntime.training.ortmodule import _io, _logger, _onnx_models, _utils
from onnxruntime.training.ortmodule.debug_options import DebugOptions, LogLevel

from .provider_options import ProviderOptions,OpenVINOProviderOptions
from . import utils
import inspect

# Needed to override PyTorch methods
T = TypeVar("T", bound="Module")

class ORTInferenceModule(torch.nn.Module):
    """Extends user's :class:`torch.nn.Module` model to leverage ONNX Runtime super fast inference engine.

    ORTInferenceModule specializes the user's :class:`torch.nn.Module` model, providing :meth:`~torch.nn.Module.forward`
    along with all other :class:`torch.nn.Module`'s APIs which are applicable for optimized inference.

    Args:
        module (torch.nn.Module): User's PyTorch module that ORTInferenceModule specializes
        debug_options (:obj:`DebugOptions`, optional): debugging options for ORTInferenceModule.
        provider_options (:obj:`ProviderOptions`, optional): execution provider options for ORTInferenceModule.
    """

    def __init__(self, module, debug_options=None, provider_options=None):
        
        if not debug_options:
            debug_options = DebugOptions()

        if not provider_options:
            provider_options = ProviderOptions()

        super(ORTInferenceModule, self).__init__()
        self._original_module = module
        utils.patch_ortinferencemodule_forward_method(self)
        self._flattened_module = _io._FlattenedModule(module)
        self._debug_options = debug_options
        # onnx models
        self._onnx_models = _onnx_models.ONNXModels()
        self._module_parameters = list(inspect.signature(self._original_module.forward).parameters.values())
        self._device = utils.get_device_from_module(module)
        self._export_mode = torch.onnx.TrainingMode.EVAL
        self._export_extra_kwargs = {}
        self._provider_options = provider_options
        self._graph_initializers = None
        self._graph_info = None
        self._inference_session = None

    def _forward_call(self, *inputs, **kwargs):
        """Delegate the :meth:`~torch.nn.Module.forward` to
        ONNX Runtime.

        The first call to forward performs setup and checking steps. During this call,
        ORTInferenceModule determines whether the module can be trained with ONNX Runtime. For
        this reason, the first forward call execution takes longer than subsequent calls.
        Execution is interrupted if ONNX Runtime cannot process the model for inferencing.

        Args:
            *inputs and **kwargs represent the positional, variable positional, keyword
            and variable keyword arguments defined in the user's PyTorch module's forward
            method. Values can be torch tensors and primitive types.

        Returns:
            The output as expected from the forward method defined by the user's
            PyTorch module. Output values supported include tensors, nested sequences
            of tensors and nested dictionaries of tensor values.
        """
        schema = _io._extract_schema({"args": copy.copy(inputs), "kwargs": copy.copy(kwargs)})
        if (
            self._onnx_models.exported_model
            and schema == self._input_info.schema
        ):
            # All required models have already been exported previously
            pass
        else:
            # Exporting module to ONNX for the first time
            self._set_device_from_module(inputs, kwargs)
            self._onnx_models.exported_model = self._export_model(schema,*inputs, **kwargs)
            if self._debug_options.save_onnx_models.save:
                self._onnx_models.save_exported_model(
                    self._debug_options.save_onnx_models.path,
                    self._debug_options.save_onnx_models.name_prefix,
                    self._export_mode,
                )

        # Create the inference_session
        if not self._inference_session:
           session_options, providers, provider_options = self._get_session_config()
           self._inference_session = onnxruntime.InferenceSession(
            self._onnx_models.exported_model.SerializeToString() , session_options, providers, provider_options
           )

        run_options = C.RunOptions()

        # Pre-process inputs to make them compatible with onnxruntime
        
        # Use IO binding
        io_binding = self._inference_session.io_binding()
        ort_inputs = []
        for inp in inputs:
            ort_inputs.append(inp)
        _utils._create_iobinding(io_binding, inputs, self._onnx_models.exported_model, self._device)
        
        # Run inference session
        self._inference_session.run_with_iobinding(io_binding, run_options)
        
        # Post-process outputs to make them compatible with pytorch 
        forward_outputs = io_binding.get_outputs()

        # forward outputs is a list (std::vector<OrtValue>) but _ortvalues_to_torch_tensor
        # is expected a OrtValueVector (also std::vector<OrtValue> but defined in onnxruntime-training).
           # _ortvalues_to_torch_tensor_list needs to be used.
        user_outputs = _utils._ortvalues_to_torch_tensor_list(forward_outputs, self._device)
        return _io.unflatten_user_output(self._module_output_schema, user_outputs)

    def _export_model(self, input_schema, *inputs, **kwargs):
        """Exports PyTorch `self._flattened_module` to ONNX for inferencing, using `*inputs` and `**kwargs` as input
        """

        # Setup dynamic axes for onnx model
        self._input_info = _io.parse_inputs_for_onnx_export(self._module_parameters, None, input_schema, inputs, kwargs)
        (
            output_names,
            output_dynamic_axes,
            self._module_output_schema,
        ) = _io.parse_outputs_for_onnx_export_and_extract_schema(self._original_module, inputs, kwargs)
        self._input_info.dynamic_axes.update(output_dynamic_axes)

        # FlattenedModule needs _InputInfo to expand user input from *args to
        # *args + **kwargs
        self._flattened_module._input_info = self._input_info

        # Export torch.nn.Module to ONNX
        f = io.BytesIO()

        # Deepcopy inputs, since input values may change after model run.
        # NOTE: Inputs may contain tensors that have attributes preventing their deepcopy (example grad_fn).
        # Therefore, deepcopy only the data component of the input tensors for
        # export.
        sample_inputs_copy, sample_kwargs_copy = _io.deepcopy_model_input(*inputs, **kwargs)
        # NOTE: Flattening the input will change the 'input schema', resulting
        # in a re-export
        sample_inputs_as_tuple = tuple(self._input_info.flatten(sample_inputs_copy, sample_kwargs_copy, self._device))

        try:
            with torch.no_grad():
                required_export_kwargs = {
                    "input_names": self._input_info.names,
                    "output_names": output_names,
                    "opset_version": ortmodule.ONNX_OPSET_VERSION,
                    "do_constant_folding": True,
                    "dynamic_axes": self._input_info.dynamic_axes,
                    "verbose": self._debug_options.logging.log_level < LogLevel.WARNING,
                    "operator_export_type": OperatorExportTypes.ONNX_ATEN_FALLBACK,
                    "export_params": True,
                    "keep_initializers_as_inputs": False,
                }

                invalid_args = self._export_extra_kwargs.keys() & required_export_kwargs.keys()
                assert (
                        len(invalid_args) == 0
                    ), f"The following PyTorch exporter arguments cannot be specified: '{invalid_args}'."
                torch.onnx.export(
                    self._flattened_module,
                    sample_inputs_as_tuple,
                    f,
                    **required_export_kwargs,
                    **self._export_extra_kwargs,
                )
        except Exception as e:
            raise RuntimeError(
                "Model could not be exported to ONNX"
            ) from e

        return onnx.load_model_from_string(f.getvalue())
      
    def _get_session_config(self):
        """Creates and returns the session configuration to be used."""

        providers = None
        provider_options = None

        # Set EP based on Provider Options
        if self._device.type == "cpu":
           if self._provider_options.provider == "openvino":
              providers = ["OpenVINOExecutionProvider"]
              providers.append("CPUExecutionProvider")
              provider_option_map = {}
              # OpenVINO EP options
              backend = self._provider_options.backend
              precision = self._provider_options.precision
              if backend and precision:
                 device_type = backend + "_" + precision
                 provider_option_map["device_type"] = device_type
           else:
              providers = ["CPUExecutionProvider"]
              provider_options = [{}]

        #set session options
        session_options = onnxruntime.SessionOptions()
        session_options.enable_mem_pattern = False
        session_options.enable_mem_reuse = False
        # default to PRIORITY_BASED execution order
        session_options.execution_order = onnxruntime.ExecutionOrder.PRIORITY_BASED
        # 0:Verbose, 1:Info, 2:Warning. 3:Error, 4:Fatal. Default is 2.
        session_options.log_severity_level = int(self._debug_options.logging.log_level)

        return session_options, providers, provider_options

    def __setattr__(self, name: str, value) -> None:

        if name in self.__dict__:
            # If the name is an attribute of ORTInferenceModule, update only ORTInferenceModule
            self.__dict__[name] = value
        else:
            # Setting any new attributes should be done on ORTInferenceModule only when 'torch_module' is not defined
            self.__dict__[name] = value

    def _set_device_from_module(self, inputs, kwargs):
        """Get the device from the module and save it to self._device"""

        device = utils.get_device_from_module(self._original_module) or utils.get_device_from_inputs(inputs, kwargs)
        if not self._device or self._device != device:
            self._device = device
            if not self._device:
                raise (
                    RuntimeError("A device must be specified in the model or inputs!")
                )

