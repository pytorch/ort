import os
import pytest
import tempfile
import torch
import torch_ort.experimental
from torch_ort import ORTModule


class NeuralNetSinglePositionalArgument(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetSinglePositionalArgument, self).__init__()

        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, input1):
        out = self.fc1(input1)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def test_save_onnx():
    # Generating a safe filename prefix to save onnx graphs
    prefix = ''
    with tempfile.NamedTemporaryFile() as f:
        prefix = f.name

    # Setting up ORTModule
    device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    model = ORTModule(model)
    torch_ort.experimental.save_intermediate_onnx_graphs(model=model, prefix=prefix)

    x = torch.randn(N, D_in, device=device)
    _ = model(x)

    # Check ONNX graphs were saved and delete them before completing the test
    success = True
    file_suffixes = ['_inference_optimized.onnx',
                     '_torch_exporter.onnx',
                     '_training.onnx',
                     '_training_optimized.onnx']
    for suffix in file_suffixes:
        filename = prefix + suffix
        if not os.path.exists(filename):
            success = False
        else:
            os.remove(filename)

    assert success is True


@pytest.mark.parametrize("bad_model", [None, bool])
def test_save_onnx_with_bad_model(bad_model):

    prefix = os.path.join(tempfile.gettempdir(), 'prefix')
    with pytest.raises(TypeError) as runtime_error:
        torch_ort.experimental.save_intermediate_onnx_graphs(model=bad_model, prefix=prefix)
    assert "`model` must be a `ORTModule` object, but " in str(runtime_error.value)


@pytest.mark.parametrize("bad_prefix, error_type", (['/sys/ortmodule/prefix', 'folder_not_exist'],
                                                    [f'{tempfile.gettempdir()}/', 'prefix_not_valid']))
def test_save_onnx_with_bad_prefix(bad_prefix, error_type):
    device = 'cuda'
    D_in, H, D_out = 784, 500, 10
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    model = ORTModule(model)

    print(bad_prefix)
    if error_type == 'folder_not_exist':
        with pytest.raises(NotADirectoryError) as runtime_error:
            torch_ort.experimental.save_intermediate_onnx_graphs(model=model, prefix=bad_prefix)
        assert "is not a valid directory to save ONNX graphs" in str(runtime_error.value)
    elif error_type == 'prefix_not_valid':
        with pytest.raises(NameError) as runtime_error:
            torch_ort.experimental.save_intermediate_onnx_graphs(model=model, prefix=bad_prefix)
        assert "is not a valid prefix name for the ONNX graph files" in str(runtime_error.value)


@pytest.mark.parametrize("level", [torch_ort.experimental.LogLevel.VERBOSE,
                                   torch_ort.experimental.LogLevel.INFO,
                                   torch_ort.experimental.LogLevel.WARNING,
                                   torch_ort.experimental.LogLevel.ERROR,
                                   torch_ort.experimental.LogLevel.FATAL])
def test_set_loglevel(level):
    # Setting up ORTModule
    device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    model = ORTModule(model)
    torch_ort.experimental.set_log_level(model=model, level=level)

    x = torch.randn(N, D_in, device=device)
    _ = model(x)

    # Check log level are really set on `model`
    for mode in [True, False]:
        assert model._execution_manager(is_training=mode)._loglevel == level


@pytest.mark.parametrize("bad_level", [None, 1])
def test_set_loglevel_with_bad_loglevel(bad_level):
    # Setting up ORTModule
    device = 'cuda'
    D_in, H, D_out = 784, 500, 10
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    model = ORTModule(model)

    with pytest.raises(TypeError) as runtime_error:
        torch_ort.experimental.set_log_level(model=model, level=bad_level)
    assert "`level` must be a `LogLevel` object, but " in str(runtime_error.value)


@pytest.mark.parametrize("bad_model", [None, bool])
def test_set_loglevel_with_bad_model(bad_model):

    with pytest.raises(TypeError) as runtime_error:
        torch_ort.experimental.set_log_level(model=bad_model, level=torch_ort.experimental.LogLevel.WARNING)
    assert "`model` must be a `ORTModule` object, but " in str(runtime_error.value)
