# Contribute to PyTorch/ORT

Internal instructions for build and upload.

## Build

1. Update version.txt

2. Install setuptools

  - `pip install setuptools`

3. Remove the current package

  - `rm dist/*`

4. Build the package

  - `python setup.py bdist_wheel`

## Publish

1. Install twine

- `pip install twine`

2. Upload to PyPI

- `twine upload dist/*`
