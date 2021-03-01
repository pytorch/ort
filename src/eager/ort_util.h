// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <core/session/onnxruntime_cxx_api.h>

#include "ort_backends.h"

// FIXME: no idea how to increase logging level for INFO, DEBUG, etc
#define ORT_LOG LOG(WARNING) << "[ORT] "

namespace torch_ort {
namespace eager {

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

template <typename T>
void CreateMLValue(onnxruntime::AllocatorPtr alloc, const std::vector<int64_t>& dims, const std::vector<T>& value,
                   OrtValue* p_mlvalue) {
  onnxruntime::TensorShape shape(dims);
  auto element_type = onnxruntime::DataTypeImpl::GetType<T>();
  std::unique_ptr<onnxruntime::Tensor> p_tensor = onnxruntime::make_unique<onnxruntime::Tensor>(element_type,
                                                                      shape,
                                                                      alloc);
  if (value.size() > 0) {
    CopyVectorToTensor(value, *p_tensor);
  }

  p_mlvalue->Init(p_tensor.release(),
                  onnxruntime::DataTypeImpl::GetType<onnxruntime::Tensor>(),
                  onnxruntime::DataTypeImpl::GetType<onnxruntime::Tensor>()->GetDeleteFunc());
}

template <typename T>
void CreateMLValue(void* data_ptr, const std::vector<int64_t>& dims, OrtValue* p_mlvalue) {
  onnxruntime::TensorShape shape(dims);
  auto element_type = onnxruntime::DataTypeImpl::GetType<T>();
  OrtMemoryInfo *cpu_info;
  Ort::GetApi().CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &cpu_info);
  std::unique_ptr<onnxruntime::Tensor> p_tensor = onnxruntime::make_unique<onnxruntime::Tensor>(element_type,
                                                                      shape,
                                                                      data_ptr,
                                                                      *cpu_info);
  
  p_mlvalue->Init(p_tensor.release(),
                  onnxruntime::DataTypeImpl::GetType<onnxruntime::Tensor>(),
                  onnxruntime::DataTypeImpl::GetType<onnxruntime::Tensor>()->GetDeleteFunc());
}

std::vector<int64_t> GetStrides(const std::vector<int64_t>& shape, int64_t element_size);

} // namespace eager
} // namespace torch_ort