import tempfile
import warnings
from pathlib import Path

import pytest
import torch
import torchvision
from torch_ort import ORTInferenceModule

warnings.filterwarnings("ignore")


class TestORTInferenceModule:
    @pytest.mark.parametrize(
        "model_name",
        ["resnet18", "resnet50"],
    )
    @pytest.mark.parametrize("pretrained", [True, False])
    def test_build_torchvision_model(self, model_name: str, pretrained: bool):
        """Test to build torchvision models and convert them to ORTInferenceModule"""
        model = getattr(torchvision.models, model_name)(pretrained=pretrained)
        assert isinstance(model, torch.nn.Module)

        model = ORTInferenceModule(model)
        assert isinstance(model, ORTInferenceModule)

    @pytest.mark.parametrize(
        "model_name",
        ["resnet18", "resnet50"],
    )
    @pytest.mark.parametrize("pretrained", [True, False])
    def test_forward_pass_torchvision_model(self, model_name: str, pretrained: bool):
        """Test forward pass of ORTInferenceModule from torchvision models"""
        model = getattr(torchvision.models, model_name)(pretrained=pretrained)
        assert isinstance(model, torch.nn.Module)

        model = ORTInferenceModule(model)
        assert isinstance(model, ORTInferenceModule)

        input = torch.rand((1, 3, 256, 256))
        pred = model(input)
        assert isinstance(pred, torch.Tensor)

    @pytest.mark.parametrize("filename", ["test"])
    def test_save_onnx_model(self, filename):
        """Test to save exported onnx model"""
        from onnxruntime.training.ortmodule.debug_options import DebugOptions

        debug_options = DebugOptions(save_onnx=True, onnx_prefix=filename)
        assert isinstance(debug_options, DebugOptions)

        model = getattr(torchvision.models, "resnet18")(pretrained=True)
        assert isinstance(model, torch.nn.Module)

        model = ORTInferenceModule(model, debug_options=debug_options)
        assert isinstance(model, ORTInferenceModule)

        # need to run forward pass or else the model isn't exported
        input = torch.rand((1, 3, 256, 256))
        pred = model(input)
        assert isinstance(pred, torch.Tensor)

        saved_model = Path(f"{filename}_torch_exported_inference.onnx")
        assert saved_model.exists()

    def test_set_verbose_log_ortinferencemodule(self):
        """Test settings the log level of ORTInferenceModule to `verbose`"""
        from onnxruntime.training.ortmodule.debug_options import (DebugOptions,
                                                                  LogLevel)

        debug_options = DebugOptions(log_level=LogLevel.VERBOSE)
        assert isinstance(debug_options, DebugOptions)

        model = getattr(torchvision.models, "resnet18")(pretrained=True)
        model = ORTInferenceModule(model, debug_options=debug_options)
        assert isinstance(model, ORTInferenceModule)

        # need to run forward pass or else the model isn't exported
        input = torch.rand((1, 3, 256, 256))
        pred = model(input)
        assert isinstance(pred, torch.Tensor)

    def test_export_model(self):
        """Test `_export_model` call"""
        model = getattr(torchvision.models, "resnet18")(pretrained=True)
        assert isinstance(model, torch.nn.Module)

        model = ORTInferenceModule(model)
        assert isinstance(model, ORTInferenceModule)
        assert model._onnx_models.exported_model == None

        input = torch.rand((1, 3, 256, 256))
        pred = model(input)
        assert isinstance(pred, torch.Tensor)
        assert model._onnx_models.exported_model != None
