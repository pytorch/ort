import setuptools
from datetime import date

with open("Readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

def fetch_requirements(path):
    with open(path, 'r') as fd:
        return [r.strip() for r in fd.readlines()]

version_str = open('version.txt', 'r').read().strip()

if 'dev' in version_str:
    version_str = version_str + date.today().strftime("%Y%m%d")

install_requires = fetch_requirements('requirements.txt')

setuptools.setup(
    name="torch_ort",
    version=version_str,
    author="torch-ort contributors",
    description="training Pytorch models with onnxruntime",
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
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
)
