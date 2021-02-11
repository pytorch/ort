#pragma once

#include <torch/extension.h>
#include "core/framework/ml_value.h"
#include "core/framework/tensor.h"

namespace torch_ort {
namespace eager {

at::Tensor new_with_orttensor_ort(OrtValue&& ot, const at::TensorOptions& options);

const OrtValue& orttensor_from_ort(const at::Tensor& tensor);

OrtValue& orttensor_from_ort(at::Tensor& tensor);

} // namespace eager
} // namespace torch_ort