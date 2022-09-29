# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import torch
from packaging.version import Version
from torch.onnx import register_custom_op_symbolic

class CustomOpSymbolicRegistry:
    _SYMBOLICS = {}

    @classmethod
    def register(cls, name, domain, fn):
        cls._SYMBOLICS[domain + "::" + name] = fn

    @classmethod
    def register_all(cls):
        for name, fn in cls._SYMBOLICS.items():
            # Symbolic name is in format: domain::name
            # Exporter will fail to register symbolic with non-empty domain when torch version is < 1.11.0.
            if Version(torch.__version__) >= Version("1.11.0") or name.startswith("::"):
                register_custom_op_symbolic(name, fn, 1)


def register_symbolic(name, domain=""):
    def symbolic_wrapper(fn):
        CustomOpSymbolicRegistry.register(name, domain, fn)
        return fn

    return symbolic_wrapper

# Unsupported Aten ops to be added here

@register_symbolic("grid_sampler")
def grid_sampler(g, self, grid, mode, padding_mode, align_corners):
    output = g.op("org.pytorch.aten::ATen", self, grid, mode, padding_mode, align_corners, operator_s="grid_sampler")
    output.setType(self.type())
    return output

@register_symbolic("triu")
def triu(g, self, diagonal, out=None):
    out = g.op("org.pytorch.aten::ATen", self, diagonal, operator_s="triu")
    out.setType(self.type())
    return out

@register_symbolic("tril")
def tril(g, self, diagonal, out=None):
    out = g.op("org.pytorch.aten::ATen", self, diagonal, operator_s="tril")
    out.setType(self.type())
    return out