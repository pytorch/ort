# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import subprocess
import sys
import getopt
import argparse

from glob import glob
import shutil
from shutil import which

parser = argparse.ArgumentParser(description='Build Torch_ORT eager')
parser.add_argument('--build_config', default='Debug', type=str,
                    choices=['Debug', 'Release'], help='Build Config')
parser.add_argument('--additional_libs', default=None, type=str, help='Additional libraries to link against')
parser.add_argument('--compiler_args', default=None, type=str, help='Additional compiler args to use')
parser.add_argument('--skip_tests', action='store_true', help='Skips running unit tests as part of the build tests')
parser.add_argument('--user', action='store_true', help='Install to user')
parser.add_argument('free_args', nargs='*')

args = parser.parse_args()

build_config = 'Debug'

if args.build_config:
  build_config = args.build_config

# replace any remaining args
sys.argv[1:] = args.free_args
if args.user:
  sys.argv += ['--user']

def is_debug_build():
  return build_config != 'Release'

python_exe = sys.executable
self_dir = os.path.dirname(os.path.realpath(__file__))
repo_root_dir = os.path.realpath(os.path.join(self_dir, '..', '..'))

pytorch_src_dir = os.path.join(
  repo_root_dir, 'external', 'pytorch')
pytorch_compile_commands_path = os.path.join(
  pytorch_src_dir, 'compile_commands.json')

if not os.path.exists(os.path.join(pytorch_src_dir, '.git')):
  raise Exception('pytorch submodule does not exist. Run git submodule update --init --recursive')

ort_src_dir = os.path.join(repo_root_dir, 'external', 'onnxruntime')
ort_build_dir = os.path.join(self_dir, 'ort_build', build_config)

if not os.path.exists(os.path.join(ort_src_dir, '.git')):
  raise Exception('onnxruntime submodule does not exist. Run git submodule update --init --recursive')

ort_lib_dirs = [
  ort_build_dir
]

import numpy as np

ort_include_dirs = [
  os.path.join(ort_src_dir, 'include', 'onnxruntime'),
  os.path.join(ort_src_dir, 'include', 'onnxruntime', 'core', 'session'),
  os.path.join(ort_src_dir, 'onnxruntime'),
  os.path.join(ort_src_dir, 'orttraining'),
  os.path.join(ort_build_dir),
  os.path.join(ort_src_dir, 'cmake', 'external', 'onnx'),
  os.path.join(ort_src_dir, 'cmake', 'external', 'SafeInt'),
  os.path.join(ort_src_dir, 'cmake', 'external', 'protobuf', 'src'),
  os.path.join(ort_src_dir, 'cmake', 'external', 'nsync', 'public'),
  os.path.join(ort_src_dir, 'cmake', 'external', 'mp11', 'include'),
  os.path.join(ort_src_dir, 'cmake', 'external', 'optional-lite', 'include'),
  os.path.join(ort_src_dir, 'cmake', 'external', 'dlpack', 'include'),
  os.path.join(ort_build_dir, 'external', 'onnx'),
  np.get_include()
]

ort_static_libs = [os.path.join(ort_build_dir, f'{l}.a') for l in [
  'libonnxruntime_eager',
  'libonnxruntime_training',
  'libonnxruntime_session',
  'libonnxruntime_providers',
  'libonnxruntime_framework',
  'libonnxruntime_optimizer',
  'libonnxruntime_util',
  'libonnxruntime_graph',
  'libonnxruntime_mlas',
  'libonnxruntime_flatbuffers',
  'libonnxruntime_common',
  'libonnxruntime_optimizer'
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
    'develop',
    '--user'
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
    '--use_mpi', 'true',
    '--build_eager_mode',
    '--enable_pybind'
  ]
  if which('ninja'):
    args += ['--cmake_generator', 'Ninja']
  print("Invoking:", str(args))
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

if args.additional_libs:
  for file in args.additional_libs.split(':'):
    print('Adding following to ort_staticlibs:', file)
    ort_static_libs.append(file)

extra_compile_args = [
  '-std=c++14',
  '-fsized-deallocation',
  '-DONNX_ML',
  '-DONNX_NAMESPACE=onnx',
  '-DENABLE_TRAINING',
  f'-DONNX_BUILD_CONFIG="{build_config}"',
]

if is_debug_build():
  extra_compile_args += [
    '-g',
    '-DONNX_DEBUG'
  ]

if args.compiler_args:
  extra_compile_args += [args.compiler_args]

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

ort_python_bind_path = os.path.join(ort_src_dir, 'onnxruntime', 'python')
ort_python_bind_training_path = os.path.join(ort_src_dir, 'orttraining', 'orttraining', 'python')

eager_src = (glob('*.cpp') + 
             [os.path.join(ort_python_bind_path, 'onnxruntime_pybind_exceptions.cc'),
             os.path.join(ort_python_bind_path, 'onnxruntime_pybind_iobinding.cc'), 
             os.path.join(ort_python_bind_path, 'onnxruntime_pybind_mlvalue.cc'),
             os.path.join(ort_python_bind_path, 'onnxruntime_pybind_ortvalue.cc'),
             os.path.join(ort_python_bind_path, 'onnxruntime_pybind_state_common.cc'),
             os.path.join(ort_python_bind_path, 'onnxruntime_pybind_state.cc'),
             os.path.join(ort_python_bind_path, 'onnxruntime_pybind_sparse_tensor.cc'),
             os.path.join(ort_python_bind_training_path, 'orttraining_pybind_state.cc'),
             os.path.join(ort_src_dir, 'orttraining', 'orttraining', 'core', 'framework', 'torch', 'dlpack_python.cc')])

setup(
  name='torch_ort',
  ext_modules=[
    CppExtension(
      name='torch_ort',
      sources=eager_src,
      extra_compile_args=extra_compile_args,
      runtime_library_dirs=[ort_build_dir],
      include_dirs=ort_include_dirs,
      library_dirs=ort_lib_dirs,
      extra_objects=ort_static_libs)
  ],
  cmdclass={
    'build_ext': BuildExtension
  })

if not args.skip_tests:
  # copy the test_provider_lib to test_data folder
  if not os.path.exists('test_data'):
    os.makedirs('test_data')
  shutil.copy2(os.path.join(ort_build_dir, 'libtest_execution_provider.so'), os.path.join(self_dir, 'test_data'))
  subprocess.check_call([
    python_exe,
    os.path.join(self_dir, 'test')
  ], cwd=os.path.join(self_dir, 'test'))
