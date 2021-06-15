// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <core/providers/cpu/cpu_execution_provider.h>
#include <core/common/logging/sinks/clog_sink.h>
#include "ort_backends.h"
#include "ort_log.h"

#ifdef USE_MSNPU
  namespace onnxruntime {
    std::unique_ptr<onnxruntime::IExecutionProvider> CreateMSNPU_ExecutionProvider();
  }
#endif

namespace torch_ort {
namespace eager {

using namespace onnxruntime;

std::unique_ptr<Environment> CreateEnvironment() {
    std::unique_ptr<Environment> env;    
    const std::string logger_id{"torch_ort"};
    auto logging_manager = std::make_unique<logging::LoggingManager>(
        std::unique_ptr<logging::ISink>{new logging::CLogSink{}},
        logging::Severity::kVERBOSE, false,
        logging::LoggingManager::InstanceType::Default,
        &logger_id); 
    Environment::Create(std::move(logging_manager), env);
    return env;
}

ORTBackendsManager& GetORTBackendsManager() {
  static auto env = CreateEnvironment();
  static ORTBackendsManager instance {env->GetLoggingManager()->DefaultLogger()};
  return instance;
}

onnxruntime::ORTInvoker& GetORTInvoker(const at::Device device) {
  return GetORTBackendsManager().GetInvoker(device);
}

onnxruntime::ORTInvoker& ORTBackendsManager::GetInvoker(const at::Device device) {
  ORT_LOG_FN(device);

  auto device_index = 0;
  if (device.has_index()) {
    device_index = device.index();
  }

  TORCH_CHECK(device.type() == at::DeviceType::ORT, "must be an ORT device");
  TORCH_CHECK(device_index >= 0, "must have a valid index");

  auto lookup = backends_.find(device.index());
  if (lookup != backends_.end()) {
    return *lookup->second;
  }

#ifdef USE_MSNPU
  auto ep = onnxruntime::CreateMSNPU_ExecutionProvider();
#else
  auto ep = std::make_unique<onnxruntime::CPUExecutionProvider>(
    onnxruntime::CPUExecutionProviderInfo(false));
#endif

  auto invoker = 
    std::make_unique<onnxruntime::ORTInvoker>(
      std::move(ep),
      logger_);

  backends_[device_index] = std::move(invoker);
  return *backends_[device_index];
}

} // namespace eager
} // namespace torch_ort