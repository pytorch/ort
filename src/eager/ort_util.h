// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <core/session/onnxruntime_cxx_api.h>

#include "ort_backends.h"

namespace torch_ort {
namespace eager {

void CreateMLValue(onnxruntime::AllocatorPtr alloc, 
                   onnxruntime::MLDataType element_type, 
                   const std::vector<int64_t>& dims, 
                   OrtValue* p_mlvalue);

void CreateMLValue(void* data_ptr, onnxruntime::MLDataType element_type, const std::vector<int64_t>& dims, OrtValue* p_mlvalue);

template <typename T>
inline void CopyVectorToTensor(const std::vector<T>& value, onnxruntime::Tensor& tensor) {
  gsl::copy(gsl::make_span(value), tensor.MutableDataAsSpan<T>());
}

// vector<bool> is specialized so we need to handle it separately
template <>
inline void CopyVectorToTensor<bool>(const std::vector<bool>& value, onnxruntime::Tensor& tensor) {
  auto output_span = tensor.MutableDataAsSpan<bool>();
  for (size_t i = 0, end = value.size(); i < end; ++i) {
    output_span[i] = value[i];
  }
}

std::vector<int64_t> GetStrides(const std::vector<int64_t>& shape, int64_t element_size);

} // namespace eager
} // namespace torch_ort