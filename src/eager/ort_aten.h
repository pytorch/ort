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

const at::Tensor get_at_tensor_from_ort_tensor(OrtValue&& ot, const at::TensorOptions& options);

const OrtValue get_ort_tensor_from_at_tensor(const at::Tensor& tensor);

const onnxruntime::MLDataType get_ort_scalar_type_from_aten(at::ScalarType dtype);

} // namespace eager
} // namespace torch_ort