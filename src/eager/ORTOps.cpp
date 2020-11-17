#include <ATen/InferSize.h>

#include "ORTOps.h"

namespace at {
namespace native {
namespace ort {
namespace detail {

ORTTensor reshape_copy(
  const ORTTensor& input,
  std::vector<int64_t> shape) {

  // TODO: actual reshape on buffer
  ORTTensor output{infer_size(shape, input.numel())};
  return output;
}

} // namespace detail
} // namespace ort
} // namespace native
} // namespace at
