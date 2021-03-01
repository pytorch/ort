// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <torch/extension.h>
#include <core/framework/ml_value.h>

#include "ort_util.h"
#include "ort_ops.h"

namespace torch_ort {
namespace eager {

at::Tensor new_with_orttensor_ort(OrtValue&& ot, const at::TensorOptions& options);

const OrtValue& orttensor_from_ort(const at::Tensor& tensor);

OrtValue& orttensor_from_ort(at::Tensor& tensor);

} // namespace eager
} // namespace torch_ort