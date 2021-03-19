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
    kCPU = 1,
    kCUDA = 2,
    kDirectML = 3
  };

  ORTBackendsManager() {
    backend_kinds_[ORTDeviceKind::kCPU] = "cpu";
    backend_kinds_[ORTDeviceKind::kCUDA] = "cuda";
    backend_kinds_[ORTDeviceKind::kDirectML] = "direct_ml";
  }

  std::map<ORTDeviceKind, std::string>& GetBackendKinds() {
    return backend_kinds_;
  }

  onnxruntime::ORTInvoker& GetInvoker(const at::Device device);

private:
  std::map<ORTDeviceKind, std::string> backend_kinds_;
  std::map<at::DeviceIndex, std::unique_ptr<onnxruntime::ORTInvoker>> backends_;
};

ORTBackendsManager& GetORTBackendsManager();

onnxruntime::ORTInvoker& GetORTInvoker(const at::Device device);

} // namespace eager
} // namespace torch_ort