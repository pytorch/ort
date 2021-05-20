import os
import pytest
import tempfile
import torch
from torch_ort.experimental import set_log_level,\
                                   LogLevel,\
                                   save_intermediate_onnx_graphs,\
                                   set_propagate_cast_ops_optimization,\
                                   PropagateCastOpsStrategy,\
                                   PropagateCastLevel
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

@pytest.mark.parametrize("enable", [True, False, None])
def test_save_onnx(enable):
    # Generating a safe filename prefix to save onnx graphs
    prefix = ''
    with tempfile.NamedTemporaryFile() as f:
        prefix = f.name

    # Setting up ORTModule
    device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    model = ORTModule(model)

    if enable is None:
        # Use implicit default value
        save_intermediate_onnx_graphs(model=model, prefix=prefix)
        # But explicitly set default value for assertion below
        enable = True
    else:
        save_intermediate_onnx_graphs(model=model, enable=enable, prefix=prefix)

    x = torch.randn(N, D_in, device=device)
    _ = model(x)

    # Check saving status
    assert enable == model._execution_manager(model._is_training())._save_onnx

    # Check ONNX graphs were saved and delete them before completing the test
    success = True
    file_suffixes = ['_inference_optimized.onnx',
                     '_torch_exporter.onnx',
                     '_training.onnx',
                     '_training_optimized.onnx']
    for suffix in file_suffixes:
        filename = prefix + suffix
        if (enable and not os.path.exists(filename)) or (not enable and os.path.exists(filename)):
            success = False

        # Clean-up time
        if os.path.exists(filename):
            os.remove(filename)

    assert success is True


@pytest.mark.parametrize("bad_model", [None, bool])
def test_save_onnx_with_bad_model(bad_model):

    prefix = os.path.join(tempfile.gettempdir(), 'prefix')
    with pytest.raises(TypeError) as runtime_error:
        save_intermediate_onnx_graphs(model=bad_model, enable=True, prefix=prefix)
    assert "`model` must be a `ORTModule` object, but " in str(runtime_error.value)


@pytest.mark.parametrize("bad_prefix, error_type", (['/sys/ortmodule/prefix', 'folder_not_exist'],
                                                    [f'{tempfile.gettempdir()}/', 'prefix_name_not_valid'],
                                                    [None, 'prefix_type_not_valid'],
                                                    [True, 'prefix_type_not_valid']))
def test_save_onnx_with_bad_prefix(bad_prefix, error_type):
    device = 'cuda'
    D_in, H, D_out = 784, 500, 10
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    model = ORTModule(model)

    if error_type == 'folder_not_exist':
        with pytest.raises(NotADirectoryError) as runtime_error:
            save_intermediate_onnx_graphs(model=model, prefix=bad_prefix)
        assert "is not a valid directory to save ONNX graphs" in str(runtime_error.value)
    elif error_type == 'prefix_name_not_valid':
        with pytest.raises(NameError) as runtime_error:
            save_intermediate_onnx_graphs(model=model, prefix=bad_prefix)
        assert "is not a valid prefix name for the ONNX graph files" in str(runtime_error.value)
    elif error_type == 'prefix_type_not_valid':
        with pytest.raises(TypeError) as runtime_error:
            save_intermediate_onnx_graphs(model=model, prefix=bad_prefix)
        assert "`prefix` must be a non-empty string" in str(runtime_error.value)

@pytest.mark.parametrize("level", [LogLevel.VERBOSE,
                                   LogLevel.INFO,
                                   LogLevel.WARNING,
                                   LogLevel.ERROR,
                                   LogLevel.FATAL,
                                   None])
def test_set_loglevel(level):
    # Setting up ORTModule
    device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    model = ORTModule(model)

    if level is None:
        # Use implicit default value
        set_log_level(model=model)
        # But explicitly set default value for assertion below
        level = LogLevel.WARNING
    else:
        set_log_level(model=model, level=level)

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
        set_log_level(model=model, level=bad_level)
    assert "`level` must be a `LogLevel` object, but " in str(runtime_error.value)


@pytest.mark.parametrize("bad_model", [None, bool])
def test_set_loglevel_with_bad_model(bad_model):

    with pytest.raises(TypeError) as runtime_error:
        set_log_level(model=bad_model, level=LogLevel.WARNING)
    assert "`model` must be a `ORTModule` object, but " in str(runtime_error.value)


@pytest.mark.parametrize("strategy, level", ([PropagateCastOpsStrategy.NONE, PropagateCastLevel.NOT_USED],
                                             [PropagateCastOpsStrategy.INSERT_AND_REDUCE, PropagateCastLevel.FASTER_KEEP_PRECISION],
                                             [PropagateCastOpsStrategy.FLOOD_FILL, PropagateCastLevel.FASTER_KEEP_PRECISION],
                                             [PropagateCastOpsStrategy.REMOVE_INPUT_OUTPUT_UP_DOWN_CASTS, PropagateCastLevel.AGGRRESSIVE_MIXED_PRECISION]))
def test_set_propagate_cast(strategy, level):
    # Setting up ORTModule
    device = 'cuda'
    D_in, H, D_out = 784, 500, 10
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    model = ORTModule(model)

    set_propagate_cast_ops_optimization(model=model, level=level, strategy=strategy)
    for mode in [True, False]:
        assert model._execution_manager(is_training=mode)._propagate_cast_ops_strategy == strategy
        assert model._execution_manager(is_training=mode)._propagate_cast_ops_level == level



@pytest.mark.parametrize("bad_model", [None, bool])
def test_set_propagate_cast_with_bad_model(bad_model):

    with pytest.raises(TypeError) as runtime_error:
        set_propagate_cast_ops_optimization(model=bad_model)
    assert "`model` must be a `ORTModule` object, but " in str(runtime_error.value)

@pytest.mark.parametrize("bad_strategy", [None, bool])
def test_set_propagate_cast_with_bad_strategy(bad_strategy):
    # Setting up ORTModule
    device = 'cuda'
    D_in, H, D_out = 784, 500, 10
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    model = ORTModule(model)

    with pytest.raises(TypeError) as runtime_error:
        set_propagate_cast_ops_optimization(model=model, strategy=bad_strategy)
    assert "`strategy` must be a `PropagateCastOpsStrategy` object, but" in str(runtime_error.value)

@pytest.mark.parametrize("bad_level", [None, -2, 3])
def test_set_propagate_cast_with_bad_level(bad_level):
    # Setting up ORTModule
    device = 'cuda'
    D_in, H, D_out = 784, 500, 10
    model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    model = ORTModule(model)

    strategy = PropagateCastOpsStrategy.INSERT_AND_REDUCE
    with pytest.raises(TypeError) as runtime_error:
        set_propagate_cast_ops_optimization(model=model, strategy=strategy, level=bad_level)
    assert "`level` must be a `PropagateCastLevel` object" in str(runtime_error.value)
