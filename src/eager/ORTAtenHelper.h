#pragma once

#include <torch/extension.h>
#include "core/framework/ml_value.h"
#include "core/framework/tensor.h"


namespace at {
namespace native {
namespace ort {
namespace aten {

Tensor new_with_orttensor_ort(OrtValue&& ot, const TensorOptions& options);

const OrtValue& orttensor_from_ort(const Tensor& tensor);

OrtValue& orttensor_from_ort(Tensor& tensor);

}
}
}
}