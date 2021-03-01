// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <core/providers/cpu/cpu_execution_provider.h>

#include "ort_backends.h"

namespace torch_ort {
namespace eager {

using namespace onnxruntime;

ORTBackendsManager& GetORTBackendsManager() {
  static ORTBackendsManager instance;
  return instance;
}

onnxruntime::ORTInvoker& GetORTInvoker(const at::Device device) {
  return GetORTBackendsManager().GetInvoker(device);
}

onnxruntime::ORTInvoker& ORTBackendsManager::GetInvoker(const at::Device device) {
  TORCH_CHECK(device.type() == at::DeviceType::ORT, "must be an ORT device");
  TORCH_CHECK(device.index() >= 0, "must have a valid index");

  auto lookup = backends_.find(device.index());
  if (lookup != backends_.end()) {
    return *lookup->second;
  }

  // FIXME: create the correct EP from (device_kind, device_index):
  ORTDeviceKind device_kind;
  uint8_t device_index;
  GetORTDeviceKindAndIndex(device.index(), device_kind, device_index);

  auto ep = onnxruntime::make_unique<onnxruntime::CPUExecutionProvider>(
    onnxruntime::CPUExecutionProviderInfo(false));

  auto invoker = 
    onnxruntime::make_unique<onnxruntime::ORTInvoker>(
      std::move(ep));

  backends_.insert({device.index(), std::move(invoker)});
  return *backends_[device.index()];
}

} // namespace eager
} // namespace torch_ort