# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import copy
import os
import pytest
import tempfile
import torch

import _test_helpers
from torch_ort import ORTModule, DebugOptions, LogLevel, set_seed
from torch_ort.optim import FusedAdam
from torch_ort.utils.data import LoadBalancingDistributedSampler, LoadBalancingDistributedBatchSampler

class NeuralNetSinglePositionalArgument(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetSinglePositionalArgument, self).__init__()

        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, input1):
        out = self.fc1(input1)
        out = self.relu(out)
        out = self.fc2(out)
        return self.dropout(out)


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)

def test_set_seed():
    N, D_in, H, D_out = 64, 784, 500, 10
    input = torch.randn(N, D_in)
    orig_model = NeuralNetSinglePositionalArgument(D_in, H, D_out)
    predictions = []
    for _ in range(10):
        set_seed(1)
        model = ORTModule(copy.deepcopy(orig_model))
        prediction = model(input)
        predictions.append(prediction)

    # All predictions must match
    for pred in predictions:
        _test_helpers.assert_values_are_close(predictions[0], pred, rtol=1e-9, atol=0.0)

@pytest.mark.parametrize("mode", ['training', 'inference'])
def test_debug_options_save_onnx_models_os_environment(mode):

    N, D_in, H, D_out = 64, 784, 500, 10
    # Create a temporary directory for the onnx_models
    with tempfile.TemporaryDirectory() as temporary_dir:
        os.environ['ORTMODULE_SAVE_ONNX_PATH'] = temporary_dir
        model = NeuralNetSinglePositionalArgument(D_in, H, D_out)
        ort_model = ORTModule(model, DebugOptions(save_onnx=True, onnx_prefix='my_model'))
        if mode == 'inference':
            ort_model.eval()
        x = torch.randn(N, D_in)
        _ = ort_model(x)

        # assert that the onnx models have been saved
        assert os.path.exists(os.path.join(temporary_dir, f"my_model_torch_exported_{mode}.onnx"))
        assert os.path.exists(os.path.join(temporary_dir, f"my_model_optimized_{mode}.onnx"))
        del os.environ['ORTMODULE_SAVE_ONNX_PATH']

@pytest.mark.parametrize("mode", ['training', 'inference'])
def test_debug_options_save_onnx_models_cwd(mode):

    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out)
    ort_model = ORTModule(model, DebugOptions(save_onnx=True, onnx_prefix='my_cwd_model'))
    if mode == 'inference':
        ort_model.eval()
    x = torch.randn(N, D_in)
    _ = ort_model(x)

    # assert that the onnx models have been saved
    assert os.path.exists(os.path.join(os.getcwd(), f"my_cwd_model_torch_exported_{mode}.onnx"))
    assert os.path.exists(os.path.join(os.getcwd(), f"my_cwd_model_optimized_pre_grad_{mode}.onnx"))
    assert os.path.exists(os.path.join(os.getcwd(), f"my_cwd_model_optimized_{mode}.onnx"))
    assert os.path.exists(os.path.join(os.getcwd(), f"my_cwd_model_execution_model_{mode}.onnx"))

    os.remove(os.path.join(os.getcwd(), f"my_cwd_model_torch_exported_{mode}.onnx"))
    os.remove(os.path.join(os.getcwd(), f"my_cwd_model_optimized_pre_grad_{mode}.onnx"))
    os.remove(os.path.join(os.getcwd(), f"my_cwd_model_optimized_{mode}.onnx"))
    os.remove(os.path.join(os.getcwd(), f"my_cwd_model_execution_model_{mode}.onnx"))

def test_debug_options_save_onnx_models_validate_fail_on_non_writable_dir():

    non_existent_directory = None
    with tempfile.TemporaryDirectory() as temporary_dir:
        non_existent_directory = temporary_dir

    os.environ['ORTMODULE_SAVE_ONNX_PATH'] = non_existent_directory
    with pytest.raises(Exception) as ex_info:
        _ = DebugOptions(save_onnx=True, onnx_prefix='my_model')
    assert f"Directory {non_existent_directory} is not writable." in str(ex_info.value)
    del os.environ['ORTMODULE_SAVE_ONNX_PATH']

def test_debug_options_save_onnx_models_validate_fail_on_non_str_prefix():
    prefix = 23
    with pytest.raises(Exception) as ex_info:
        _ = DebugOptions(save_onnx=True, onnx_prefix=prefix)
    assert f"Expected name prefix of type str, got {type(prefix)}." in str(ex_info.value)

def test_debug_options_save_onnx_models_validate_fail_on_no_prefix():
    with pytest.raises(Exception) as ex_info:
        _ = DebugOptions(save_onnx=True)
    assert f"onnx_prefix must be provided when save_onnx is set." in str(ex_info.value)

def test_debug_options_log_level():
    # NOTE: This test will output verbose logging

    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out)
    ort_model = ORTModule(model, DebugOptions(log_level=LogLevel.VERBOSE))
    x = torch.randn(N, D_in)
    _ = ort_model(x)

    # assert that the logging is done in verbose mode
    assert ort_model._torch_module._execution_manager(True)._debug_options.logging.log_level == LogLevel.VERBOSE

def test_debug_options_log_level_os_environment():
    # NOTE: This test will output info logging

    os.environ['ORTMODULE_LOG_LEVEL'] = 'INFO'
    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out)
    ort_model = ORTModule(model)
    x = torch.randn(N, D_in)
    _ = ort_model(x)

    # assert that the logging is done in info mode
    assert ort_model._torch_module._execution_manager(True)._debug_options.logging.log_level == LogLevel.INFO
    del os.environ['ORTMODULE_LOG_LEVEL']

def test_debug_options_log_level_validation_fails_on_type_mismatch():
    log_level = 'some_string'
    with pytest.raises(Exception) as ex_info:
        _ = DebugOptions(log_level=log_level)
    assert f"Expected log_level of type LogLevel, got {type(log_level)}." in str(ex_info.value)

def test_fused_adam_import():
    device = "cuda"
    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)

    ort_fused_adam_optimizer = FusedAdam(
        model.parameters(), lr=1e-3, weight_decay=0.01, eps=1e-8
    )

    def run_step(model, x):
        prediction = model(x)
        loss = prediction.sum()
        loss.backward()

        return loss

    def run_optim_step(optimizer):
        optimizer.step()
        optimizer.zero_grad()

    x1 = torch.randn(N, D_in, device=device, dtype=torch.float32)
    run_step(model, x1)

    run_optim_step(ort_fused_adam_optimizer)


def test_load_balancing_data_sampler_balances_import():
    samples_and_complexities = [(torch.FloatTensor([val]), torch.randint(0, 100, (1,)).item()) for val in range(100)]
    dataset = MyDataset(samples_and_complexities)

    def complexity_fn(sample):
        return sample[1]

    data_sampler = LoadBalancingDistributedSampler(
        dataset, complexity_fn=complexity_fn, world_size=2, rank=0, shuffle=False
    )

    batch_size = 12

    def batch_fn(indices):
        nonlocal batch_size
        batches = []
        for batch_index_begin in range(0, len(indices), batch_size):
            batch_index_end = min(batch_index_begin + batch_size, len(indices))
            batches.append(indices[batch_index_begin:batch_index_end])
        return batches

    batch_sampler = LoadBalancingDistributedBatchSampler(data_sampler, batch_fn)
