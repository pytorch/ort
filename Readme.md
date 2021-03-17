# training pytorch models with onnxruntime

## to build (you need to update version number in version.txt in order to be able to upload python whl):
    rm dist/*
    python setup.py bdist_wheel

## to publish:
    twine upload dist/*

## to install:
### stable:
    (uninstall onnxruntime version that do not support gpu and training)
    pip install torchort onnxruntime=1.9.0+cu111_training

### nightly:
    (make sure you are not in the ort repo folder - otherwise torchort is taken as already installed)
    pip install --pre torchort ort-gpu-nightly-training

    (eventually we are aiming at: pip install --pre torchort onnxruntime=1.9.0+cu111_training)

## to test:
    python ./ort/tests/bert_for_sequence_classification.py
## to use torchort within PyTorch training scripts:
    import onnxruntime
    from torchort import ORTModule
    model = ORTModule(model)
    # normal PyTorch training script follows
