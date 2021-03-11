# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
from glob import glob

import os
import platform
import subprocess
import sys

python_exe = sys.executable
build_config = 'Release'

def is_debug_build():
    return build_config != 'Release'

def build_ort(ort_path, build_dir):
    if not os.path.exists(build_dir):
        os.mkdir(build_dir)
    args = [python_exe, os.path.join(ort_path, 'tools', 'ci_build', 'build.py'),
            '--build_dir', build_dir, '--config', build_config,
            '--skip_submodule_sync', '--build', '--update', '--parallel']
    if platform.system() == 'Darwin':
        args += ['--cmake_generator', 'Ninja']
    subprocess.check_call(args)

def gen_ort_aten_ops():
    gen_cpp_name = "ort_aten.g.cpp"
    if os.path.exists(gen_cpp_name):
        os.remove(gen_cpp_name)
    args = [python_exe, os.path.join(os.path.dirname(__file__), 'opgen', 'opgen.py'),
             gen_cpp_name]
    subprocess.check_call(args)

build_ort('onnxruntime', 'ort_build')

current_path = os.path.abspath(os.getcwd())
ort_build_root = os.path.join(current_path, 'ort_build', build_config)
ort_lib_dir=[ort_build_root]
ort_include_dir=[os.path.join(current_path, 'onnxruntime', 'include', 'onnxruntime'),
                 os.path.join(current_path, 'onnxruntime', 'onnxruntime'),
                 os.path.join(current_path, 'ort_build', build_config),
                 os.path.join(current_path, 'onnxruntime', 'cmake', 'external', 'onnx'),
                 os.path.join(current_path, 'onnxruntime', 'cmake', 'external', 'SafeInt'),
                 os.path.join(current_path, 'onnxruntime', 'cmake', 'external', 'protobuf', 'src'),
                 os.path.join(current_path, 'onnxruntime', 'cmake', 'external', 'nsync', 'public'),
                 os.path.join(current_path, 'ort_build', build_config, 'external', 'onnx')]

ort_libs = [
                   'libonnxruntime_eager',
                   'libonnxruntime_session',
                   'libonnxruntime_providers',
                   'libonnxruntime_framework',
                   'libonnxruntime_optimizer',
                   'libonnxruntime_util',
                   'libonnxruntime_graph',
                   'libonnxruntime_mlas',
                   'libonnxruntime_flatbuffers',
                   'libonnxruntime_common', 
                   ] 
ort_static_libs = [f'{ort_build_root}/{l}.a' for l in ort_libs]

protobuf_lib = 'libprotobufd.a' if is_debug_build() else 'libprotobuf.a'

external_libs = [ort_build_root + '/external/nsync/libnsync_cpp.a',
              ort_build_root + '/external/onnx/libonnx.a',
              ort_build_root + '/external/onnx/libonnx_proto.a',
              ort_build_root + f'/external/protobuf/cmake/{protobuf_lib}',
              ort_build_root + '/external/re2/libre2.a',]
ort_static_libs.extend(external_libs)

gen_ort_aten_ops()

extra_compile_args = [
    '-std=c++14',
    '-fsized-deallocation',
    '-DONNX_ML',
    '-DONNX_NAMESPACE=onnx',
    f'-DONNX_BUILD_CONFIG="{build_config}"',
]

if is_debug_build():
    extra_compile_args += [
        '-g',
        '-DONNX_DEBUG'
    ]

setup(
    name='torch_ort',
    ext_modules=[
        CppExtension(
            name='torch_ort',
            sources=glob('*.cpp'),
            extra_compile_args=extra_compile_args,
            include_dirs=ort_include_dir,
            library_dirs=ort_lib_dir,
            extra_objects=ort_static_libs)
    ],
    cmdclass={
        'build_ext': BuildExtension
    })