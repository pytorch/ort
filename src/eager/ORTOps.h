#pragma once

#include "core/framework/ml_value.h"
#include "core/framework/tensor.h"
#include "core/eager/ort_kernel_invoker.h"

namespace at {
namespace native {
namespace ort {
namespace detail {

OrtValue reshape_copy(
  onnxruntime::ORTInvoker& invoker,
  const OrtValue& input,
  std::vector<int64_t> shape);

OrtValue add(onnxruntime::ORTInvoker& invoker,
             const OrtValue& A,
             const OrtValue& B);

void copy(onnxruntime::ORTInvoker& invoker, 
          const OrtValue& src, OrtValue& dst);

} // namespace detail
} // namespace ort
} // namespace native
} // namespace at