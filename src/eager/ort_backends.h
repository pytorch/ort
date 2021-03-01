// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <torch/extension.h>
#include <core/framework/ml_value.h>
#include <core/eager/ort_kernel_invoker.h>
#include <core/providers/cpu/cpu_execution_provider.h>

namespace torch_ort {
namespace eager {

class ORTBackendsManager {
public:
  enum ORTDeviceKind : uint8_t {
    kCPU = 0,
    kApollo = 1
  };

  ORTBackendsManager() {
    backend_kinds_[ORTDeviceKind::kCPU] = "cpu";
    backend_kinds_[ORTDeviceKind::kApollo] = "apollo";
  }

  std::map<ORTDeviceKind, std::string>& GetBackendKinds() {
    return backend_kinds_;
  }

  static inline at::Device MakePyTorchDevice(
    ORTDeviceKind device_kind,
    uint8_t device_index) {
    return at::Device(
      at::DeviceType::ORT,
      MakePyTorchDeviceIndex(
        device_kind,
        device_index));
  }

  onnxruntime::ORTInvoker& GetInvoker(const at::Device device);

private:
  std::map<ORTDeviceKind, std::string> backend_kinds_;
  std::map<at::DeviceIndex, std::unique_ptr<onnxruntime::ORTInvoker>> backends_;

  // PyTorch's DeviceIndex is a 16-bit integer into which we encode the
  // ORTDeviceKind in the high 8 bits and the ORT device index in the
  // lower 8 bits.

  static inline void GetORTDeviceKindAndIndex(
    at::DeviceIndex pytorch_device_index,
    ORTDeviceKind& device_kind,
    uint8_t& device_index) {
    device_kind = (ORTDeviceKind)(pytorch_device_index >> 8);
    device_index = (uint8_t)pytorch_device_index;
  }

  static inline at::DeviceIndex MakePyTorchDeviceIndex(
    ORTDeviceKind device_kind,
    uint8_t device_index) {
    return (uint8_t)device_kind << 8 | device_index;
  }
};

ORTBackendsManager& GetORTBackendsManager();

onnxruntime::ORTInvoker& GetORTInvoker(const at::Device device);

} // namespace eager
} // namespace torch_ort