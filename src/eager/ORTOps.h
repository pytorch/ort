#pragma once

#include "core/framework/ml_value.h"
#include "core/framework/tensor.h"

namespace at {
namespace native {
namespace ort {
namespace detail {

OrtValue reshape_copy(
  const OrtValue& input,
  std::vector<int64_t> shape);

OrtValue add(const OrtValue& A,
             const OrtValue& B);

} // namespace detail
} // namespace ort
} // namespace native
} // namespace at