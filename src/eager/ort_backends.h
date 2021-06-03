// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <torch/extension.h>
#include <core/framework/ml_value.h>
#include <core/eager/ort_kernel_invoker.h>

namespace torch_ort {
namespace eager {

class ORTBackendsManager {
public:
  ORTBackendsManager(const onnxruntime::logging::Logger& logger) : logger_(logger) {
  }

  onnxruntime::ORTInvoker& GetInvoker(const at::Device device);

private:
  std::map<at::DeviceIndex, std::unique_ptr<onnxruntime::ORTInvoker>> backends_;
  const onnxruntime::logging::Logger& logger_;
};

ORTBackendsManager& GetORTBackendsManager();

onnxruntime::ORTInvoker& GetORTInvoker(const at::Device device);

} // namespace eager
} // namespace torch_ort