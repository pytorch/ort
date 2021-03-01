// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <core/providers/cpu/cpu_execution_provider.h>

#include "ort_util.h"
#include "ort_backends.h"

namespace torch_ort {
namespace eager {

using namespace onnxruntime;

std::vector<int64_t> GetStrides(const std::vector<int64_t>& shape, int64_t element_size){
  std::vector<int64_t> strides;
  if (shape.empty())
    return strides;
  strides.resize(shape.size());
  int64_t stride = element_size;
  for (int i = shape.size() - 1; i >= 0; i--){
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}

} // namespace eager
} // namespace torch_ort