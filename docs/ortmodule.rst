ORTModule
=========

The ORTModule class uses the ONNX Runtime to accelerator PyTorch model training. 

ORTModule wraps a torch.nn.Module. It offloads the forward and backward pass of
a PyTorch training loop to ONNX Runtime. ONNX Runtime uses its optimized computation
graph and memory usage to execute these components of the training loop faster with
less memory usage.

The following code example illustrates the use of ORTModule in the simple case
where the entire model is trained using ONNX Runtime:

.. code-block:: python

    # Original PyTorch model
    class NeuralNet(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            ...
        def forward(self, x): 
            ...

    model = NeuralNet(input_size=784, hidden_size=500, num_classes=10)
    model = ORTModule(model) # Only change to original PyTorch script
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    # Training Loop is unchanged
    for data, target in data_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

ONNX Runtime can also be used to train parts of the model, by wrapping internal
torch.nn.Modules with ORTModule.

ORTModule API
-------------

.. autoclass:: torch_ort.ORTModule
    :members:
    :member-order: bysource
