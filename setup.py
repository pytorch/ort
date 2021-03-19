import setuptools
from datetime import date

with open("Readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

def fetch_requirements(path):
    with open(path, 'r') as fd:
        return [r.strip() for r in fd.readlines()]

# 1.2.0.dev1+hg.5.b11e5e6f0b0b
version_str = open('version.txt', 'r').read().strip()

if 'dev' in version_str:
    version_str = version_str + date.today().strftime("%Y%m%d")

setuptools.setup(
    name="torchort-poc",
    version=version_str,
    author="torchort contributors",
    description="training Pytorch models with onnxruntime",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pytorch/ort",
    project_urls={
        "Bug Tracker": "https://github.com/pytorch/ort/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
)
