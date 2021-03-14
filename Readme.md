# training pytorch models with onnxruntime

## to build (you need to update version number in version.txt in order to be able to upload python whl):
    rm dist/*
    python setup.py bdist_wheel

## to publish:
    twine upload dist/*

## to install:
### stable:
    pip install torch-ort

### nightly:
    pip install --pre torch-ort

## to use torch_ort within PyTorch training scripts:
    import onnxruntime
    from torch_ort import ORTModule
    model = ORTModule(model)
    # normal PyTorch training script follows
