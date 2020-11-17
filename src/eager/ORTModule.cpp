#include <torch/extension.h>

#include "ORTUtil.h"

#include "core/eager/ort_kernel_invoker.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include <memory>


namespace at {
namespace native {
namespace ort {
namespace detail {

using namespace onnxruntime;

PYBIND11_MODULE(torch_ort, m) {
  ORT_LOG << "pybind11 module init";
  std::unique_ptr<IExecutionProvider> cpu_execution_provider = onnxruntime::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo(false));
  ORTInvoker kernel_invoker(std::move(cpu_execution_provider));

  ORT_LOG << "Finish ORT kernel invoke test code";
}

} // namespace detail
} // namespace ort
} // namespace native
} // namespace at
