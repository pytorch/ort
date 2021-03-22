# training pytorch models with onnxruntime

### torch-ort can be used to train pytorch models with onnxruntime backend. Before using torch-ort, one need to have a working pytorch gpu environment.

## to build (you need to update version number in version.txt in order to be able to upload a python whl):
    (run 'pip install setuptools', if it is not already installed) 
    rm dist/*
    python setup.py bdist_wheel

## to publish (it will ask for user name and password):
    twine upload dist/*

## to install:

### stable:
    TBD
### nightly:
    (make sure you are not in the ort repo folder - otherwise torch-ort is taken as already installed)
    pip install --pre torch-ort ort-gpu-nightly-training

    (eventually we are aiming at: pip install --pre torch-ort onnxruntime=1.9.0+cu111_training)

## to test:
    python ./ort/tests/bert_for_sequence_classification.py
## to use torch-ort within PyTorch training scripts:
    import onnxruntime
    from torch_ort import ORTModule
    model = ORTModule(model)
    # normal PyTorch training script follows

## FAQs
### Question: When running training script with ORTModule, I got an error of missing Cuda shared library:
### Answer: It is possible that your torch install has a different version of Cuda library than onnxruntime. In this case, do:
    conda install -c anaconda cudatoolkit=10.2.89
    (replace with the right cudatoolkit version)