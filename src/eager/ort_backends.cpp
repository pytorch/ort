#include "ort_backends.h"

#include "core/eager/ort_kernel_invoker.h"
#include "core/providers/cpu/cpu_execution_provider.h"

namespace torch_ort {
namespace eager {

using namespace onnxruntime;

ORTBackendsManager::ORTBackendsManager(){
  // hardcode to add 1 cpu EP
  auto cpu_ep = onnxruntime::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo(false));
  backends_.push_back(std::move(onnxruntime::make_unique<onnxruntime::ORTInvoker>(std::move(cpu_ep))));
  ort_device_indices_.insert({{ORTDeviceKind::kCPU, 0}, backends_.size() - 1});
  // add 1 fake apollo EP
  auto apollo_ep = onnxruntime::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo(false));
  backends_.push_back(std::move(onnxruntime::make_unique<onnxruntime::ORTInvoker>(std::move(apollo_ep))));
  ort_device_indices_.insert({{ORTDeviceKind::kApollo, 0}, backends_.size() - 1});
}

onnxruntime::ORTInvoker& ORTBackendsManager::GetInvoker(const at::Device device){
  size_t index = device.index() < 0 ? 0 : device.index();
  assert(device.type() == DeviceType::ORT && index < backends_.size());
  return *backends_[index];
}

int ORTBackendsManager::GetPytorchDeviceIndex(ORTDeviceKind devkind, int index){
  auto it = ort_device_indices_.find({devkind, index});
  if (it == ort_device_indices_.end())
    return -1;
  return it->second;
}

} // namespace eager
} // namespace torch_ort