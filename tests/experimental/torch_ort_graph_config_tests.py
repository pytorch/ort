import pytest
import torch
from torch_ort.experimental.graph_config import set_propagate_cast_ops_optimization,\
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
        assert model._torch_module._execution_manager(is_training=mode)._propagate_cast_ops_strategy == strategy
        assert model._torch_module._execution_manager(is_training=mode)._propagate_cast_ops_level == level



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
