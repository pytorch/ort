// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <torch/extension.h>
#include <core/framework/ml_value.h>

#include "ort_util.h"
#include "ort_ops.h"
#include "ort_log.h"

namespace torch_ort {
namespace eager {

const at::Tensor aten_tensor_from_ort(
  OrtValue&& ot,
  const at::TensorOptions& options);

const onnxruntime::MLDataType ort_scalar_type_from_aten(
  at::ScalarType dtype);

const OrtValue create_ort_value(
  onnxruntime::ORTInvoker& invoker,
  const at::Scalar& scalar);

const OrtValue create_ort_value(
  onnxruntime::ORTInvoker& invoker,
  const at::Tensor& tensor);

template<typename T>
const OrtValue create_ort_value(
  onnxruntime::ORTInvoker& invoker, 
  const std::vector<T> values) {
  OrtValue ort_value;
  CreateMLValue(
    invoker.GetCurrentExecutionProvider().GetAllocator(0, OrtMemTypeDefault),
    onnxruntime::DataTypeImpl::GetType<T>(),
    {(int64_t)values.size(),},
    &ort_value);
  CopyVectorToTensor<T>(
    values,
    *ort_value.GetMutable<onnxruntime::Tensor>());
  return ort_value;
}

template<typename T>
const OrtValue create_ort_value(
  onnxruntime::ORTInvoker& invoker, 
  const at::ArrayRef<T> values) {
  std::vector<T> values_vector;
  values_vector.assign(values.begin(), values.end());
  return create_ort_value(invoker, values_vector);
}

const onnx::AttributeProto create_ort_attribute(
  const char* name,
  at::Scalar value);

} // namespace eager
} // namespace torch_ort