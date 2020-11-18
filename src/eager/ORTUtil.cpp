#include "ORTUtil.h"

#include "core/eager/ort_kernel_invoker.h"
#include "core/providers/cpu/cpu_execution_provider.h"

namespace at {
namespace native {
namespace ort {
namespace detail {

using namespace onnxruntime;

onnxruntime::ORTInvoker& GetORTInvoker(){
  static onnxruntime::ORTInvoker kernel_invoker(std::move(onnxruntime::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo(false))));
  return kernel_invoker;
}

std::vector<int64_t> GetStride(const std::vector<int64_t>& shape, int64_t element_size){
  std::vector<int64_t> strides;
  if (shape.empty())
    return strides;
  strides.resize(shape.size());
  int64_t stride = element_size;
  for (int i = shape.size() - 1; i >= 0; i--){
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}

} // namespace detail
} // namespace ort
} // namespace native
} // namespace at
