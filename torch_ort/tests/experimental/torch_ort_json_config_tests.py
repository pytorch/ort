import os

import torch
from onnxruntime.capi import _pybind_state as C
from torch_ort import ORTModule
from torch_ort.experimental.json_config import load_from_json


class Net(torch.nn.Module):
    def __init__(self, input_size=784, hidden_size=500, num_classes=10):
        super(Net, self).__init__()

        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, input1):
        out = self.fc1(input1)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def test_load_config_from_json_1():
    model = ORTModule(Net())

    # load from json once.
    path_to_json = os.path.join(os.getcwd(), "tests/experimental/torch_ort_json_config_2.json")
    load_from_json(model, path_to_json)

    # load from json another time
    path_to_json = os.path.join(os.getcwd(), "tests/experimental/torch_ort_json_config_1.json")
    load_from_json(model, path_to_json)

    for training_mode in [True, False]:
        ort_model_attributes = model._torch_module._execution_manager(training_mode)

        # test propagate cast ops
        assert (
            ort_model_attributes._runtime_options.propagate_cast_ops_strategy == C.PropagateCastOpsStrategy.FLOOD_FILL
        )
        assert ort_model_attributes._runtime_options.propagate_cast_ops_level == 3
        assert ort_model_attributes._runtime_options.propagate_cast_ops_allow == ["ABC", "DEF"]

        # test use external gpu allocator
        assert ort_model_attributes._runtime_options.use_external_gpu_allocator == False

        # test enable custom autograd function
        assert ort_model_attributes._runtime_options.enable_custom_autograd_function == True

        # test use static shape
        assert ort_model_attributes._runtime_options.use_static_shape == True

        # test run symbolic shape inference
        assert ort_model_attributes._runtime_options.run_symbolic_shape_infer == False

        # test enable grad acc optimization
        assert ort_model_attributes._runtime_options.enable_grad_acc_optimization == True

        # test skip check
        assert ort_model_attributes._runtime_options.skip_check.value == 14

        # test debug options
        assert ort_model_attributes._debug_options.save_onnx_models.save == True
        assert ort_model_attributes._debug_options.save_onnx_models.name_prefix == "my_model"
        assert ort_model_attributes._debug_options.logging.log_level.name == "VERBOSE"


def test_load_config_from_json_2():
    model = ORTModule(Net())

    # load from json once.
    path_to_json = os.path.join(os.getcwd(), "tests/experimental/torch_ort_json_config_1.json")
    load_from_json(model, path_to_json)

    # load from json another time
    path_to_json = os.path.join(os.getcwd(), "tests/experimental/torch_ort_json_config_2.json")
    load_from_json(model, path_to_json)

    for training_mode in [True, False]:
        ort_model_attributes = model._torch_module._execution_manager(training_mode)

        # test propagate cast ops
        assert (
            ort_model_attributes._runtime_options.propagate_cast_ops_strategy
            == C.PropagateCastOpsStrategy.INSERT_AND_REDUCE
        )
        assert ort_model_attributes._runtime_options.propagate_cast_ops_level == 5
        assert ort_model_attributes._runtime_options.propagate_cast_ops_allow == ["XYZ", "PQR"]

        # test use external gpu allocator
        assert ort_model_attributes._runtime_options.use_external_gpu_allocator == True

        # test enable custom autograd function
        assert ort_model_attributes._runtime_options.enable_custom_autograd_function == False

        # test use static shape
        assert ort_model_attributes._runtime_options.use_static_shape == False

        # test run symbolic shape inference
        assert ort_model_attributes._runtime_options.run_symbolic_shape_infer == True

        # test enable grad acc optimization
        assert ort_model_attributes._runtime_options.enable_grad_acc_optimization == False

        # test skip check
        assert ort_model_attributes._runtime_options.skip_check.value == 10

        # test debug options
        assert ort_model_attributes._debug_options.save_onnx_models.save == True
        assert ort_model_attributes._debug_options.save_onnx_models.name_prefix == "my_other_model"
        assert ort_model_attributes._debug_options.logging.log_level.name == "INFO"
