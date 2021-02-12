#pragma once

#include <torch/extension.h>
#include <core/framework/ml_value.h>
#include <core/eager/ort_kernel_invoker.h>

namespace torch_ort {
namespace eager {

class ORTBackendsManager{
public:

  enum ORTDeviceKind : int{
    kCPU = 0,
    kApollo = 1
  };
  ORTBackendsManager();
  onnxruntime::ORTInvoker& GetInvoker(const at::Device device);
  int GetPytorchDeviceIndex(ORTDeviceKind devkind, int index);

private:
  std::vector<std::unique_ptr<onnxruntime::ORTInvoker> > backends_;
  std::map<std::pair<ORTDeviceKind, int>, size_t> ort_device_indices_;

};

ORTBackendsManager& GetORTBackends();
onnxruntime::ORTInvoker& GetORTInvoker(at::Device device);

} // namespace eager
} // namespace torch_ort