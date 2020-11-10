from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='torch_ort',
      ext_modules=[cpp_extension.CppExtension('torch_ort', ['ORTBackend.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})



