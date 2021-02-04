#!/usr/bin/env python
#
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import sys
from typing import TextIO
import json
import opgen.lexer
import opgen.parser
from opgen.ast import *

regdecs_path = os.path.realpath(os.path.join(
  os.path.dirname(__file__),
  '..',
  '..',
  'build',
  'aten',
  'src',
  'ATen',
  'RegistrationDeclarations.h'))

onnx_ops_config_path = os.path.realpath(os.path.join(
  os.path.dirname(__file__),
  'onnx_ops.config'))

def generate_includes(writer):
  writer.write('#include <torch/extension.h> \n')
  writer.write('#include "ORTUtil.h" \n')
  writer.write('#include "ORTTensorImpl.h" \n')
  writer.write('#include "ORTOps.h" \n')
  writer.write('#include "ORTAtenHelper.h" \n')

def begin_namespace(writer):
  writer.write('namespace at { \n')
  writer.write('namespace native { \n')
  writer.write('namespace ort { \n')
  writer.write('namespace aten { \n')

def using_stmts(writer):
  writer.write('using namespace at::native::ort::detail; \n')

def end_namespace(writer):
  writer.write('}\n')
  writer.write('}\n')
  writer.write('}\n')
  writer.write('}\n')

def load_config(path):
  result = {}
  with open(path) as f:
    s = f.readline()
    while s:
      op = json.loads(s)
      result[op["aten_op"]] = op
      s = f.readline()
  return result

def gen_onnx_handle(writer, op_config, cpp_func):
  # logging
  writer.write('ORT_LOG << "%s"; \n' % cpp_func.ort_name)
  # get ort kernel invoker
  assert(len(cpp_func.parameters) > 0)
  writer.write("auto& invoker = GetORTInvoker(")
  writer.write(cpp_func.parameters[0].member.identifier.value)
  writer.write('.device()); \n')
  # prepare inputs
  i = 0
  inputs = []
  for index in op_config["inputs"]:
    writer.write('auto& input_%d = orttensor_from_ort(%s); \n' % 
                 (i, cpp_func.parameters[index].member.identifier.value))
    inputs.append('input_%d' % i)
    i += 1
  

  # prepare ort outputs:
  ort_output_name = 'results'
  # todo: how to handle multiple result
  writer.write('std::vector<OrtValue> %s(1); \n' % ort_output_name)
  # invoke ort kernel
  # todo: handle attributes
  writer.write('auto status = invoker.Invoke("%s", {%s}, %s, nullptr);\n' % 
                (op_config["onnx_op"], ','.join(inputs), ort_output_name))
  # status check
  writer.write('if (!status.IsOK()) \n')
  writer.write('  throw std::runtime_error("ORT return failure status: " + status.ErrorMessage()); \n')
  # logging
  writer.write('ORT_LOG << "Invoke ORT %s kernel successfully";\n' % op_config['onnx_op'])
  # construct result
  ort_value_name = 'ort_result'
  writer.write('OrtValue %s = %s[0]; \n' % (ort_value_name, ort_output_name))
  # todo: handle mutliple result
  # todo: return type check
  writer.write('return new_with_orttensor_ort(std::move(%s), %s.options()); \n' %
               (ort_value_name, cpp_func.parameters[0].member.identifier.value))

def write_func_signature(writer, cpp_func):
  cpp_func.return_type.write(writer)
  writer.write(" ")
  writer.write(cpp_func.ort_name)
  writer.write("(")
  for param_list_member in cpp_func.parameters:
    param_list_member.write(writer)
  writer.write(")")

if len(sys.argv) == 2:
  writer = open(sys.argv[1], 'wt')
else:
  writer = sys.stdout

with opgen.parser.cpp_create_from_file(regdecs_path) as parser:
  op_configs = load_config(onnx_ops_config_path)

  generate_includes(writer)
  begin_namespace(writer)
  using_stmts(writer)
  generated_funcs = []

  # Parse the C++ declarations
  tu = parser.parse_translation_unit()

  for cpp_func in tu:
    # Parse the Torch schema from the JSON comment that follows the C++ decl
    torch_func = None
    if cpp_func.semicolon and cpp_func.semicolon.trailing_trivia:
      for trivia in cpp_func.semicolon.trailing_trivia:
        if trivia.kind == opgen.lexer.TokenKind.SINGLE_LINE_COMMENT:
          metadata = json.loads(trivia.value.lstrip("//"))
          schema = metadata["schema"]
          schema_parser = opgen.parser.torch_create_from_string(schema)
          schema_parser.set_source_location(cpp_func.semicolon.location)
          torch_func = schema_parser.parse_function()
          torch_func.torch_schema = schema
          torch_func.torch_dispatch = metadata["dispatch"] == "True"
          torch_func.torch_default = metadata["default"] == "True"
    if not torch_func:
      continue
    
    if torch_func.identifier.value not in op_configs:
      continue

    op_config = op_configs[torch_func.identifier.value]

    cpp_func.ort_name = "ort_" + torch_func.identifier.value.replace("::", "_").replace(".", "_")
    if 'ort_aten_rename' in cpp_func.ort_name:
      continue

    writer.write(f"// {torch_func.torch_schema}\n")

    if op_config["handle"] == "Customize":
      # write the extern func declaration:
      writer.write("extern ")
      write_func_signature(writer, cpp_func)
      writer.write(";\n")
    else:
      # Write the C++ signature for our ORT implementation
      writer.write("static ")
      write_func_signature(writer, cpp_func)
      writer.write("\n{\n")


      writer.write("    //  Return: ")
      torch_func.return_type.write(writer)
      writer.write("\n")

      # Do something for each parameter; cpp_func and torch_func should have
      # 1:1 parameters, but the metadata on the torch_func is richer than
      # the cpp_func version.
      for i, param_list_member in enumerate(torch_func.parameters):
        writer.write(f"    // Param {i}: ")
        param_list_member.member.write(writer)
        writer.write("\n")

      if op_config["handle"] == "Generated":
        gen_onnx_handle(writer, op_config, cpp_func)
      else:
        writer.write('throw std::runtime_error("Not Implemented");')
      # Finalize the ORT implementation
      writer.write("}\n\n")

    generated_funcs.append((cpp_func, torch_func))

  # Generate registrations
  writer.write("TORCH_LIBRARY_IMPL(aten, ORT, m) {\n")
  writer.write("    ORT_LOG << \"ATen init\";\n")
  for cpp_func, torch_func in generated_funcs:
    if torch_func.identifier.value not in op_configs:
      continue
    op_config = op_configs[torch_func.identifier.value]

    if op_config["unbox_registration"]:
      writer.write(f"    m.impl_UNBOXED(\"{torch_func.identifier.value}\", ({cpp_func.ort_name}));\n")
    else:
      writer.write(f"    m.impl(\"{torch_func.identifier.value}\", TORCH_FN({cpp_func.ort_name}));\n")
  writer.write("}\n")

  end_namespace(writer)

writer.close()