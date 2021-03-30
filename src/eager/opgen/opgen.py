#!/usr/bin/env python3

from opgen.generator import \
  ORTGen as ORTGen, \
  ONNXOp as ONNXOp, \
  ONNXType as ONNXType, \
  ONNXAttr as ONNXAttr, \
  SignatureOnly as SignatureOnly

kMSDomain = 'onnxruntime::kMSDomain'

class Add(ONNXOp):
  def __init__(self, a, b): super().__init__('Add', 1, a, b)

class Sub(ONNXOp):
  def __init__(self, a, b): super().__init__('Sub', 1, a, b)

class Mul(ONNXOp):
  def __init__(self, a, b): super().__init__('Mul', 1, a, b)

class Gemm(ONNXOp):
  def __init__(self, a, b, c, Alpha=None, Beta=None, TransA=None, TransB=None):
    super().__init__('Gemm', 1, a, b, c,
      Alpha=ONNXAttr(Alpha, ONNXType.FLOAT),
      Beta=ONNXAttr(Beta, ONNXType.FLOAT),
      TransA=ONNXAttr(TransA, ONNXType.INT),
      TransB=ONNXAttr(TransB, ONNXType.INT))

class Transpose(ONNXOp):
  def __init__(self, data): super().__init__('Transpose', 1, data)

class Relu(ONNXOp):
  def __init__(self, x): super().__init__('Relu', 1, x)

class SingleArg(ONNXOp):
  def __init__(self, name, x): super().__init__(name, 1, x)
    
class MatMul(ONNXOp):
  def __init__(self, a, b): super().__init__('MatMul', 1, a, b)

class ReduceSum(ONNXOp):
  def __init__(self, data, axes, KeepDims=None, noop_with_empty_axes=None):
    super().__init__('ReduceSum', 1, data, axes,
      KeepDims=ONNXAttr(KeepDims, ONNXType.INT),
      noop_with_empty_axes=ONNXAttr(noop_with_empty_axes, ONNXType.INT))

class ReluGrad(ONNXOp):
  def __init__(self, dY, X):
    super().__init__('ReluGrad', 1, dY, X)
    self.domain = kMSDomain

single_arg_op_names = ["data", "_shape_as_tensor", "abs", "absolute", "angle", "sgn",
"_conj", "acos", "arccos", "acosh", "arccosh", "asinh", "arcsinh",
"atanh", "arctanh", "asin", "arcsin", "atan", "arctan", "atleast_1d",
"atleast_2d", "atleast_3d", "bitwise_not", "logical_not", "ceil",
"cos", "cosh", "erf", "erfc", "exp", "exp2", "expm1", "floor",
"frac", "inverse", "_inverse_helper", "isnan", "isreal", "log",
"log10", "log1p", "log2", "logdet", "matrix_exp", "median",
"nanmedian", "rad2deg", "deg2rad", "reciprocal", "neg", "negative",
"round", "relu", "gelu", "rsqrt", "selu", "silu", "sigmoid", "sin",
"sinh", "sqrt", "square", "tan", "tanh", "fliplr", "flipud", "trunc",
"fix", "_sparse_sum", "frobenius_norm", "to_dense", "coalesce",
"to_sparse", "to_mkldnn",  "q_per_channel_scales",
"q_per_channel_zero_points", "int_repr", "trace", "nonzero", "lgamma",
"digamma", "erfinv", "i0", "sign", "signbit", "min", "max", "all",
"any", "hardsigmoid", "hardswish", "log_sigmoid", "isfinite",
"isinf", "isposinf", "isneginf", "linalg_det", "det"]

explicit_single_arg_ops = {'aten::' + x: SingleArg(x.capitalize(), 'self') for x in single_arg_op_names}

single_arg_op_names_inplace = ["abs_", "absolute_", 
# "sgn_":  Tried to set a manually boxed kernel for a kernel that already has a boxed kernel set.
 "acos_", "arccos_", "acosh_", "arccosh_",
 "asinh_", "arcsinh_", "atanh_", "arctanh_", "asin_", "arcsin_",
"atan_", "arctan_", "bitwise_not_", "logical_not_", "ceil_", "cos_",
"cosh_", "erf_", "erfc_", "exp_", "exp2_", "expm1_", "floor_",
"frac_", "log_", "log10_", "log1p_", "log2_", "rad2deg_", "deg2rad_",
"reciprocal_", "neg_", "negative_", "round_", "relu_", "rsqrt_",
"selu_", "silu_", "sigmoid_", "sin_", "sinh_", "detach_", "squeeze_",
"sqrt_", "square_", "t_", "tan_", "tanh_", "trunc_", "fix_",
# "zero_", Odd, why is this done with SignatureOnly? above, discuss with abock
"set_", "lgamma_", "digamma_", "erfinv_", "i0_", "sign_", "hardsigmoid_",
"hardswish_"]

implicit_single_arg_ops = {'aten::' + x: SingleArg(x.capitalize()[:-1], 'self') for x in single_arg_op_names_inplace}

ops = {
  # Hand-Implemented Ops
  'aten::empty.memory_format': SignatureOnly(),
  'aten::empty_strided': SignatureOnly(),
  'aten::zeros_like': SignatureOnly(),
  'aten::zero_': SignatureOnly(),
  'aten::copy_': SignatureOnly(),
  'aten::reshape': SignatureOnly(),
  'aten::view': SignatureOnly(),

  # Fully Generated Ops
  'aten::add.Tensor': Add('self', Mul('alpha', 'other')),
  'aten::add_.Tensor': Add('self', Mul('alpha', 'other')),
  'aten::sub.Tensor': Sub('self', Mul('alpha', 'other')),
  'aten::sub_.Tensor': Sub('self', Mul('alpha', 'other')),
  'aten::mul.Tensor': Mul('self', 'other'),
  'aten::addmm': Gemm('mat1', 'mat2', 'self', Alpha='alpha', Beta='beta'),
  'aten::t': Transpose('self'),
  'aten::relu': Relu('self'),
  'aten::mm': MatMul('self', 'mat2'),
  
  'aten::sum.dim_IntList': ReduceSum('self', 'dim', KeepDims='keepdim'),
  'aten::threshold_backward': ReluGrad('grad_output', 'self')  
}
ops.update (implicit_single_arg_ops)
ops.update (explicit_single_arg_ops)
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