from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
from glob import glob

setup(
    name='torch_ort',
    ext_modules=[
        CppExtension(
            name='torch_ort',
            sources=glob('*.cpp'))
    ],
    cmdclass={
        'build_ext': BuildExtension
    })