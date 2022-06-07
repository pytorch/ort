# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import setuptools
from datetime import date
import os

with open("pypi-readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

def fetch_requirements(path):
    with open(path, 'r') as fd:
        return [r.strip() for r in fd.readlines()]

version_str = open('version.txt', 'r').read().strip()

if 'dev' in version_str:
    version_str = version_str + date.today().strftime("%Y%m%d")

# Create a new file called _version.py where the version is stored in a variable __version__
with open(os.path.join("torch_ort", "_version.py"), 'w') as fh:
    fh.write(f"__version__ = '{version_str}'")

install_requires = fetch_requirements('requirements.txt')

packages=[*setuptools.find_packages(), 'torch_ort.configure']

setuptools.setup(
    name="torch_ort_inference",
    version=version_str,
    author="torch-ort contributors",
    description="Accelerate PyTorch models with ONNX Runtime OpenVINO EP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pytorch/ort",
    install_requires=install_requires,
    project_urls={
        "Bug Tracker": "https://github.com/pytorch/ort/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=packages,
    python_requires=">=3.6",
)
