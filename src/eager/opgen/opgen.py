#!/usr/bin/env python
#
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import sys
import json

from typing import TextIO

import opgen.lexer
import opgen.parser
import opgen.writer
from opgen.ast import *

class OpMap:
  def __init__(self, torch_identifier: str,
    ort_identifier: str = None,
    signature_only: bool = False,
    unboxed: bool = False,
    params: [str] = None):
    self.torch_identifier = torch_identifier
    self.ort_identifier = ort_identifier
    self.signature_only = signature_only
    self.unboxed = unboxed
    self.params = params

op_maps = [
  OpMap(
    'aten::empty.memory_format',
    signature_only = True,
    unboxed = True),
  OpMap(
    'aten::copy_',
    signature_only = True),
  OpMap(
    'aten::reshape',
    signature_only = True),
  OpMap(
    'aten::view',
    signature_only = True),
  OpMap('aten::add.Tensor', 'Add'),
  OpMap('aten::mul.Tensor', 'Mul'),
  OpMap('aten::relu', 'Relu'),
  OpMap('aten::sub.Tensor', 'Sub')
]

op_maps = { op_map.torch_identifier: op_map for op_map in op_maps }

regdecs_path = os.path.realpath(os.path.join(
  os.path.dirname(__file__),
  '..',
  '..',
  'build',
  'aten',
  'src',
  'ATen',
  'RegistrationDeclarations.h'))

def parse_function_decls(parser: opgen.parser.CPPParser):
  # Parse the C++ declarations
  tu = parser.parse_translation_unit()

  # Parse the Torch schema from the JSON comment that follows each C++ decl
  # and link associated Torch and C++ decls (functions, parameters, returns)
  for cpp_func in tu:
    if cpp_func.semicolon and cpp_func.semicolon.trailing_trivia:
      for trivia in cpp_func.semicolon.trailing_trivia:
        if trivia.kind == opgen.lexer.TokenKind.SINGLE_LINE_COMMENT:
          metadata = json.loads(trivia.value.lstrip('//'))
          schema = metadata['schema']

          schema_parser = opgen.parser.torch_create_from_string(schema)
          schema_parser.set_source_location(cpp_func.semicolon.location)
          torch_func = schema_parser.parse_function()

          torch_func.torch_schema = schema
          torch_func.torch_dispatch = metadata['dispatch'] == 'True'
          torch_func.torch_default = metadata['default'] == 'True'

          cpp_func.torch_func = torch_func

          if cpp_func.return_type:
            cpp_func.return_type.torch_type = torch_func.return_type

          # Synthesize KWArgsSentinelType in the C++ declaration if we have one
          for i, torch_param in enumerate([p.member for p in torch_func.parameters]):
            if isinstance(torch_param.parameter_type, KWArgsSentinelType):
              cpp_func.parameters.members.insert(i, SyntaxListMember(
                torch_param,
                Token(None, TokenKind.COMMA, ',')))
              break

          # Link Torch parameters to their C++ counterparts, special casing
          # TensorOptions parameters
          for i, cpp_param in enumerate([p.member for p in cpp_func.parameters]):
            if not getattr(cpp_param, 'torch_param', None):
              cpp_param.torch_param = []

            torch_param_range = 1
            if isinstance(cpp_param.parameter_type.desugar(), TensorOptionsType):
              torch_param_range = 4

            for j in range(torch_param_range):
              torch_param = torch_func.parameters[i + j].member
              cpp_param.torch_param.append(torch_param.parameter_type.desugar())

          yield cpp_func, torch_func
          break

def write_func_signature(writer, cpp_func):
  cpp_func.return_type.write(writer)
  writer.write(' ')
  writer.write(cpp_func.ort_name)
  writer.write('(')
  writer.push_indent()
  for param_list_member in cpp_func.parameters:
    writer.writeline()
    if isinstance(param_list_member.member.parameter_type, KWArgsSentinelType):
      writer.write('// ')
    param_list_member.write(writer)
  writer.pop_indent()
  writer.write(')')

def write_func_body(writer, op_map, cpp_func):
  writer.writeline(f'ORT_LOG << "{cpp_func.ort_name}";')
  writer.writeline()

  assert(len(cpp_func.parameters) > 0)

  have_invoker = False

  # Prepare ORT kernel input parameters
  inputs = []
  for param in cpp_func.parameters:
    if isinstance(param.member.parameter_type, KWArgsSentinelType):
      break

    torch_param_name = param.member.identifier.value
    ort_param_name = f'ort_in_{torch_param_name}'
    inputs.append(ort_param_name)

    if not have_invoker:
      writer.write('auto& invoker = GetORTInvoker(')
      writer.write(torch_param_name)
      writer.writeline('.device());')
      writer.writeline()
      have_invoker = True

    writer.write('auto& ')
    writer.write(ort_param_name)
    writer.write(' = orttensor_from_ort(')
    writer.write(torch_param_name)
    writer.writeline(');')

  writer.writeline()

  # Prepare ORT results; TODO: handle multiple results
  ort_output_name = 'ort_out'
  writer.writeline(f'std::vector<OrtValue> {ort_output_name}(1);')
  writer.writeline()

  # Invoke ORT kernel; TODO: handle attributes
  writer.writeline('auto status = invoker.Invoke(')
  writer.push_indent()
  writer.writeline(f'"{op_map.ort_identifier}", {{')
  writer.push_indent()
  for i, n in enumerate(inputs):
    if i > 0:
      writer.writeline(', ')
    writer.write(n)
  writer.pop_indent()
  writer.writeline()
  writer.writeline(f'}}, {ort_output_name}, nullptr);')
  writer.pop_indent()
  writer.writeline()

  # Assert Invocation
  writer.writeline('if (!status.IsOK())')
  writer.push_indent()
  writer.writeline('throw std::runtime_error(')
  writer.push_indent()
  writer.writeline('"ORT return failure status:" + status.ErrorMessage());')
  writer.pop_indent()
  writer.pop_indent()
  writer.writeline()

  writer.writeline(f'ORT_LOG << "Invoked ORT {op_map.ort_identifier}";')
  writer.writeline()

  ort_value_name = 'ort_result'
  # TODO: pick the right "out" Torch parameter; do not assume the first one
  ort_out_param_name = cpp_func.parameters[0].member.identifier.value
  writer.writeline(f'OrtValue {ort_value_name} = {ort_output_name}[0];')

  # TODO: Handle mutliple results
  # TODO: Assert return type
  writer.writeline('return new_with_orttensor_ort(')
  writer.push_indent()
  writer.writeline(f'std::move({ort_value_name}),')
  writer.writeline(f'{ort_out_param_name}.options());')
  writer.pop_indent()

with opgen.parser.cpp_create_from_file(regdecs_path) as parser, \
  opgen.writer.SourceWriter(
    open(sys.argv[1], 'wt') \
      if len(sys.argv) == 2 \
      else sys.stdout) as writer:

  # File preamble
  writer.writeline('// AUTO-GENERATED CODE! - DO NOT EDIT!')
  writer.writeline(f'// $ python {" ".join(sys.argv)}')
  writer.writeline()
  writer.writeline('#include <torch/extension.h>')
  writer.writeline()
  writer.writeline('#include "ort_tensor.h"')
  writer.writeline('#include "ort_aten.h"')
  writer.writeline()
  writer.push_namespace('torch_ort')
  writer.push_namespace('eager')
  writer.writeline()
  writer.push_indent()
  writer.writeline('using namespace at;')
  writer.writeline()

  generated_funcs = []

  # Generate the ops we care about
  for cpp_func, torch_func in parse_function_decls(parser):
    if torch_func.identifier.value not in op_maps:
      continue

    op_map = op_maps[torch_func.identifier.value]

    cpp_func.ort_name = 'ort_' + torch_func.identifier.value \
      .replace('::', '_') \
      .replace('.', '_')

    writer.writeline(f'// {torch_func.torch_schema}')

    if op_map.signature_only:
      # write the extern func declaration:
      writer.write('extern ')
      write_func_signature(writer, cpp_func)
      writer.writeline(';')
    else:
      # Write the C++ signature for our ORT implementation
      writer.write('static ')
      write_func_signature(writer, cpp_func)
      writer.writeline(' {')
      writer.push_indent()

      write_func_body(writer, op_map, cpp_func)

      # Finalize the ORT implementation
      writer.pop_indent()
      writer.writeline('}')

    writer.writeline()

    generated_funcs.append((cpp_func, torch_func))

  # Generate registrations
  writer.writeline('TORCH_LIBRARY_IMPL(aten, ORT, m) {')
  writer.push_indent()
  writer.writeline('ORT_LOG << "ATen init";')

  for cpp_func, torch_func in generated_funcs:
    if torch_func.identifier.value not in op_maps:
      continue

    op_map = op_maps[torch_func.identifier.value]

    if op_map.unboxed:
      reg_method = 'impl_UNBOXED'
      fn_wrapper = ''
    else:
      reg_method = 'impl'
      fn_wrapper = 'TORCH_FN'

    writer.write(f'm.{reg_method}("{torch_func.identifier.value}", ')
    writer.writeline(f'{fn_wrapper}({cpp_func.ort_name}));')

  writer.pop_indent()
  writer.writeline('}')

  # Finalize the file
  writer.pop_indent()
  writer.pop_namespace()
  writer.pop_namespace()