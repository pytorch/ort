#pragma once


#include "core/framework/ml_value.h"
#include "core/framework/tensor.h"
#include "core/eager/ort_kernel_invoker.h"
#include <torch/extension.h>

#include "core/session/onnxruntime_cxx_api.h"

namespace at {
namespace native {
namespace ort {
namespace detail {

class ORTBackendsManager{
public:

  enum ORTDeviceKind : int{
    kCPU = 0,
    kApollo = 1
  };
  ORTBackendsManager();
  onnxruntime::ORTInvoker& GetInvoker(const Device device);
  int GetPytorchDeviceIndex(ORTDeviceKind devkind, int index);

private:
  std::vector<std::unique_ptr<onnxruntime::ORTInvoker> > backends_;
  std::map<std::pair<ORTDeviceKind, int>, size_t> ort_device_indices_;

};

} // namespace detail
} // namespace ort
} // namespace native
} // namespace at