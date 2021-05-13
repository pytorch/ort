# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import subprocess
import sys

from glob import glob
from shutil import which

build_config = 'Debug'
def is_debug_build():
  return build_config != 'Release'

python_exe = sys.executable
self_dir = os.path.dirname(os.path.realpath(__file__))
repo_root_dir = os.path.realpath(os.path.join(self_dir, '..', '..'))

pytorch_src_dir = os.path.join(
  repo_root_dir, 'external', 'pytorch')
pytorch_compile_commands_path = os.path.join(
  pytorch_src_dir, 'compile_commands.json')

ort_src_dir = os.path.join(repo_root_dir, 'external', 'onnxruntime')
ort_build_dir = os.path.join(self_dir, 'ort_build', build_config)

ort_lib_dirs = [
  ort_build_dir
]

ort_include_dirs = [
  os.path.join(ort_src_dir, 'include', 'onnxruntime'),
  os.path.join(ort_src_dir, 'onnxruntime'),
  os.path.join(ort_build_dir),
  os.path.join(ort_src_dir, 'cmake', 'external', 'onnx'),
  os.path.join(ort_src_dir, 'cmake', 'external', 'SafeInt'),
  os.path.join(ort_src_dir, 'cmake', 'external', 'protobuf', 'src'),
  os.path.join(ort_src_dir, 'cmake', 'external', 'nsync', 'public'),
  os.path.join(ort_src_dir, 'cmake', 'external', 'mp11', 'include'),
  os.path.join(ort_build_dir, 'external', 'onnx')
]

ort_static_libs = [os.path.join(ort_build_dir, f'{l}.a') for l in [
  'libonnxruntime_eager',
  'libonnxruntime_session',
  'libonnxruntime_providers',
  'libonnxruntime_framework',
  'libonnxruntime_optimizer',
  'libonnxruntime_util',
  'libonnxruntime_graph',
  'libonnxruntime_mlas',
  'libonnxruntime_flatbuffers',
  'libonnxruntime_common'
]] + [
  os.path.join(ort_build_dir, 'external', 'nsync', 'libnsync_cpp.a'),
  os.path.join(ort_build_dir, 'external', 'onnx', 'libonnx.a'),
  os.path.join(ort_build_dir, 'external', 'onnx', 'libonnx_proto.a'),
  os.path.join(ort_build_dir, 'external', 'protobuf', 'cmake',
    'libprotobufd.a' if is_debug_build() else 'libprotobuf.a'),
  os.path.join(ort_build_dir, 'external', 're2', 'libre2.a'),
  os.path.join(ort_build_dir, 'tensorboard', 'libtensorboard.a')
]

def build_pytorch():
  env = dict(os.environ)
  if is_debug_build():
    env['DEBUG'] = '1'
  subprocess.check_call([
    python_exe,
    'setup.py',
    'develop'
  ], cwd=pytorch_src_dir, env=env)

def build_ort():
  if not os.path.exists(ort_build_dir):
    os.makedirs(ort_build_dir)
  args = [
    python_exe,
    os.path.join(ort_src_dir, 'tools', 'ci_build', 'build.py'),
    '--build_dir', os.path.dirname(ort_build_dir),
    '--config', build_config,
    '--skip_submodule_sync',
    '--build',
    '--update',
    '--parallel',
    '--enable_training',
    '--disable_nccl',
    '--use_mpi', 'false'
  ]
  if which('ninja'):
    args += ['--cmake_generator', 'Ninja']
  subprocess.check_call(args)

def gen_ort_aten_ops():
  gen_cpp_name = "ort_aten.g.cpp"
  if os.path.exists(gen_cpp_name):
    os.remove(gen_cpp_name)
  subprocess.check_call([
    python_exe,
    os.path.join(self_dir, 'opgen', 'opgen.py'),
    gen_cpp_name
  ])

if os.path.isfile(pytorch_compile_commands_path):
  print('Skipping PyTorch Build (remove compile_commands.json to build it):')
  print(f'  {pytorch_compile_commands_path}')
else:
  print('Building PyTorch...')
  build_pytorch()

build_ort()
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

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
  name='torch_ort',
  ext_modules=[
    CppExtension(
      name='torch_ort',
      sources=glob('*.cpp'),
      extra_compile_args=extra_compile_args,
      include_dirs=ort_include_dirs,
      library_dirs=ort_lib_dirs,
      extra_objects=ort_static_libs)
  ],
  cmdclass={
    'build_ext': BuildExtension
  })

subprocess.check_call([
    python_exe,
    os.path.join(
        os.path.dirname(__file__),
        'test',
        'ort_ops.py')])