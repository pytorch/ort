#pragma once

#include "ORTTensor.h"

namespace at {
namespace native {
namespace ort {
namespace detail {

ORTTensor reshape_copy(
  const ORTTensor& input,
  std::vector<int64_t> shape);

} // namespace detail
} // namespace ort
} // namespace native
} // namespace at