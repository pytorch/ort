# ONNX Runtime Eager Mode Support for PyTorch

## Dependencies & Environment

### Ubuntu 20.04

  * Run the [`provision-ubuntu-20.04.sh`](provision-ubuntu-20.04.sh) script.

## Build

### Ensure Submodules are Current

```bash
git submodule update --recursive --init
```

### Build All Components

Within this `/src/eager` directory, the `setup.py` script can drive the initial
bootstrapping build of all components:

* PyTorch (`/external/pytorch`)
* ONNX Runtime (`/external/onnxruntime`)
* Eager Mode Extension (`/src/eager`)

```bash
python setup.py develop --user
```