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

at::Tensor new_with_orttensor_ort(OrtValue&& ot, const at::TensorOptions& options);

const OrtValue& orttensor_from_ort(const at::Tensor& tensor);

OrtValue& orttensor_from_ort(at::Tensor& tensor);

onnxruntime::MLDataType get_ort_scalar_type_from_aten(at::ScalarType dtype);

} // namespace eager
} // namespace torch_ort