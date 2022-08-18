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

    # Need to add logic to check if iGPU is present and make this skip conditional
    @pytest.mark.skip
    @pytest.mark.parametrize(
        "backend",
        ["GPU"],
    )
    @pytest.mark.parametrize(
        "precision",
        ["FP32", "FP16"],
    )
    def test_openvino_provider_GPU(self, backend: str, precision: str):
        """Test ORTInferenceModule with GPU backend with FP32 | FP16 precision"""
        provider_options = OpenVINOProviderOptions(backend=backend, precision=precision)
        assert isinstance(provider_options, OpenVINOProviderOptions)
        model = getattr(torchvision.models, "resnet18")(pretrained=True)
        assert isinstance(model, torch.nn.Module)

        model = ORTInferenceModule(model, provider_options=provider_options)
        assert isinstance(model, ORTInferenceModule)

        input = torch.rand((1, 3, 256, 256))
        pred = model(input)
        assert isinstance(pred, torch.Tensor)

    # Need to add logic to check if VPU is present and make this skip conditional
    @pytest.mark.skip
    def test_openvino_provider_VPU(self):
        """Test ORTInferenceModule with VPU backend with FP16 precision"""
        provider_options = OpenVINOProviderOptions(backend="MYRIAD", precision="FP16")
        assert isinstance(provider_options, OpenVINOProviderOptions)
        model = getattr(torchvision.models, "resnet18")(pretrained=True)
        assert isinstance(model, torch.nn.Module)

        model = ORTInferenceModule(model, provider_options=provider_options)
        assert isinstance(model, ORTInferenceModule)

        input = torch.rand((1, 3, 256, 256))
        pred = model(input)
        assert isinstance(pred, torch.Tensor)
