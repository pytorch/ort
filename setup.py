import setuptools

with open("Readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

def fetch_requirements(path):
    with open(path, 'r') as fd:
        return [r.strip() for r in fd.readlines()]

install_requires = fetch_requirements('./requirements.txt')

version_str = open('version.txt', 'r').read().strip()

setuptools.setup(
    name="torch_ort-poc",
    version=version_str,
    author="torch_ort contributors",
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
    install_requires=install_requires,
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
)
