# ONNX Runtime Eager Mode Support

## Dependencies & Environment

```bash
conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses pkg-config libuv flake8 llvm-openmp
```

```bash
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
```

## Build

Run the following:
```
git submodule update --init --recursive
```

From this folder (src/eager) build the torch_ort extension:
```bash
python setup.py install
```

