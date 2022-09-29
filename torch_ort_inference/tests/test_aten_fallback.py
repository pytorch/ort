import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_ort import ORTInferenceModule, DebugOptions, LogLevel

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        mask = torch.triu(torch.ones_like(x))
        x = x * mask
        return x

class TestAtenFallback:
    def test_aten_fb_custom_model(self):
        var = torch.ones(1, 1, 224, 224)
        model = Model()
        debug_options = DebugOptions(save_onnx=True, onnx_prefix='atenfb', log_level=LogLevel.VERBOSE)
        model = ORTInferenceModule(model, debug_options=debug_options)
        output = model(var)
        assert isinstance(output, torch.Tensor)