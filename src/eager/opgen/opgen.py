#!/usr/bin/env python3
#
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from copy import deepcopy

from opgen.generator import \
  ORTGen as ORTGen, \
  ONNXOp as ONNXOp, \
  SignatureOnly as SignatureOnly, \
  MakeFallthrough as MakeFallthrough

from opgen.onnxops import *

kMSDomain = 'onnxruntime::kMSDomain'

class ReluGrad(ONNXOp):
  def __init__(self, dY, X):
    super().__init__('ReluGrad', 1, dY, X)
    self.domain = kMSDomain

ops = {
  # Hand-Implemented Ops
  'aten::empty.memory_format': SignatureOnly(),
  'aten::empty_strided': SignatureOnly(),
  'aten::zeros_like': SignatureOnly(),
  'aten::zero_': SignatureOnly(),
  'aten::copy_': SignatureOnly(),
  'aten::reshape': SignatureOnly(),
  'aten::view': SignatureOnly(),

  'aten::addmm': Gemm('mat1', 'mat2', 'self', alpha='alpha', beta='beta'),
  'aten::t': Transpose('self'),
  'aten::mm': MatMul('self', 'mat2'),
  'aten::hardshrink': Shrink('self', bias=0, 'lambd'),
  'aten::softshrink': Shrink('self', 'bias', 'lambd'),
  'aten::sum.dim_IntList': ReduceSum('self', 'dim', KeepDims='keepdim'),
  'aten::threshold_backward': ReluGrad('grad_output', 'self'),

  'aten::fmod.Scalar': Mod('self', 'other', fmod=1),
  'aten::fmod.Tensor': Mod('self', 'other', fmod=1),
}

for binary_op, onnx_op in {
  'add': Add('self', Mul('alpha', 'other')),
  'sub': Sub('self', Mul('alpha', 'other')),
  'mul': Mul('self', 'other'),
  'div': Div('self', 'other')}.items():
  for dtype in ['Tensor', 'Scalar']:
    for variant in ['', '_']:
      ops[f'aten::{binary_op}{variant}.{dtype}'] = deepcopy(onnx_op)

for unary_op in [
  'abs','acos','acosh', 'asinh', 'atanh', 'asin', 'atan', 'ceil', 'cos',
  'cosh', 'erf', 'exp', 'floor', 'isnan', 'log', 'reciprocal', 'neg', 'round',
  'relu', 'selu', 'sigmoid', 'sin', 'sinh', 'sqrt', 'tan', 'tanh', 'nonzero',
  'sign', 'min', 'max', 'hardsigmoid', 'isinf', 'det']:
  aten_name = f'aten::{unary_op}'
  onnx_op = onnx_ops[unary_op]('self')
  ops[aten_name] = onnx_op
  # produce the in-place variant as well for ops that support it
  if unary_op not in ['isnan', 'nonzero', 'min', 'max', 'isinf', 'det']:
    ops[f'{aten_name}_'] = onnx_op

ortgen = ORTGen(ops)

import os
import sys

from opgen.parser import cpp_create_from_file as CPPParser
from opgen.writer import SourceWriter as SourceWriter

regdecs_path = os.path.realpath(os.path.join(
  os.path.dirname(__file__),
  '..',
  '..',
  'build',
  'aten',
  'src',
  'ATen',
  'RegistrationDeclarations.h'))
print(regdecs_path)
output = sys.stdout
if len(sys.argv) >= 2:
  output = open(sys.argv[1], 'wt')

with CPPParser(regdecs_path) as parser, SourceWriter(output) as writer:
  ortgen.run(parser, writer)