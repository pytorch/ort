# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

def main():
    from onnxruntime.training.ortmodule.torch_cpp_extensions import install as ortmodule_install
    ortmodule_install.build_torch_cpp_extensions()

if __name__ == '__main__':
    main()
