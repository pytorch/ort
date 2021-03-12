training pytorch models with onnxruntime

## to build (you need to increase version minor number in version.txt in order to upload python whl):
    rm dist/*
    python setup.py bdist_wheel

## to publish:
    python -m twine upload --repository testpypi dist/*

## to install:
    pip install -i https://test.pypi.org/simple/ torch-ort-poc

## to use torch_ort within PyTorch training scripts:
    import onnxruntime
    from torch_ort import ORTModule
    model = ORTModule(model)
    # normal PyTorch training script follows
