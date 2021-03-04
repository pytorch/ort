// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <core/providers/cpu/cpu_execution_provider.h>

#include "ort_util.h"
#include "ort_backends.h"

namespace torch_ort {
namespace eager {

using namespace onnxruntime;


void CreateMLValue(onnxruntime::AllocatorPtr alloc, 
                   onnxruntime::MLDataType element_type, 
                   const std::vector<int64_t>& dims, 
                   OrtValue* p_mlvalue) {
  onnxruntime::TensorShape shape(dims);
  std::unique_ptr<onnxruntime::Tensor> p_tensor = onnxruntime::make_unique<onnxruntime::Tensor>(element_type,
                                                                      shape,
                                                                      alloc);
  p_mlvalue->Init(p_tensor.release(),
                  onnxruntime::DataTypeImpl::GetType<onnxruntime::Tensor>(),
                  onnxruntime::DataTypeImpl::GetType<onnxruntime::Tensor>()->GetDeleteFunc());
}

void CreateMLValue(void* data_ptr, onnxruntime::MLDataType element_type, const std::vector<int64_t>& dims, OrtValue* p_mlvalue) {
  onnxruntime::TensorShape shape(dims);
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