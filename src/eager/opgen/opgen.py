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

with opgen.parser.cpp_create_from_file(regdecs_path) as parser:
  writer = sys.stdout
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
          schema_parser = opgen.parser.torch_create_from_string(
            metadata["schema"])
          torch_func = schema_parser.parse_function()
          torch_func.torch_dispatch = metadata["dispatch"] == "True"
          torch_func.torch_default = metadata["default"] == "True"
    if not torch_func:
      continue

    cpp_func.ort_name = "ort_" + torch_func.identifier.value.replace("::", "_")

    # Write the C++ signature for our ORT implementation
    writer.write("static ")
    cpp_func.return_type.write(writer)
    writer.write(" ")
    writer.write(cpp_func.ort_name)
    writer.write("(")
    for param_list_member in cpp_func.parameters:
      param_list_member.write(writer)
    writer.write(")\n{\n")

    # Do something for each parameter; cpp_func and torch_func should have
    # 1:1 parameters, but the metadata on the torch_func is richer than
    # the cpp_func version.
    for i, param_list_member in enumerate(torch_func.parameters):
      writer.write(f"    // Param {i}: ")
      param_list_member.member.write(writer)
      writer.write("\n")

    # Finalize the ORT implementation
    writer.write("}\n\n")

    generated_funcs.append((cpp_func, torch_func))

  # Generate registrations
  writer.write("TORCH_LIBRARY_IMPL(aten, ORT, m) {\n")
  writer.write("    ORT_LOG << \"ATen init\";\n")
  for cpp_func, torch_func in generated_funcs:
    writer.write(f"    m.impl(\"{torch_func.identifier.value}\", TORCH_FN({cpp_func.ort_name}));\n")
  writer.write("}\n")