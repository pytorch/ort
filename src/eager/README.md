# ONNX Runtime Eager Mode Support

## Dependencies & Environment

```bash
conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses pkg-config libuv
```

```bash
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
```

## Build

Build pytorch first from root of this repo:

```bash
python setup.py install
```

Then go to this folder and build the torch_ort extension:
```bash
python setup.py install
```

