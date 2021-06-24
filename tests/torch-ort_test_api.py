# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import _test_helpers

import copy
import onnxruntime as ort
import torch

from onnxruntime.training import ORTModule


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

def test_set_seed():
    device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10
    input = torch.randn(N, D_in, device=device)
    orig_model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    predictions = []
    for _ in range(10):
        ort.set_seed(1)
        model = ORTModule(copy.deepcopy(orig_model))
        prediction = model(input)
        predictions.append(prediction)

    # All predictions must match
    for pred in predictions:
        _test_helpers.assert_values_are_close(predictions[0], pred, rtol=1e-9, atol=0.0)
