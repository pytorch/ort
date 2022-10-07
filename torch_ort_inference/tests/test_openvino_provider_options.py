import pytest
import torch
import torchvision
from torch_ort import OpenVINOProviderOptions, ORTInferenceModule

class TestOpenVinoProviderOptions:
    def test_openvino_provider_CPU(self):
        """Test ORTInferenceModule with CPU backend with FP32 precision"""
        provider_options = OpenVINOProviderOptions(backend="CPU", precision="FP32")
        assert isinstance(provider_options, OpenVINOProviderOptions)
        model = getattr(torchvision.models, "resnet18")(pretrained=True)
        assert isinstance(model, torch.nn.Module)

        model = ORTInferenceModule(model, provider_options=provider_options)
        assert isinstance(model, ORTInferenceModule)

        input = torch.rand((1, 3, 256, 256))
        pred = model(input)
        assert isinstance(pred, torch.Tensor)
