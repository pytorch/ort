// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <torch/extension.h>

#include "ort_backends.h"
#include "ort_log.h"

namespace torch_ort {
namespace eager {

PYBIND11_MODULE(torch_ort, torch_ort_module) {
  ORT_LOG_DEBUG << "pybind11 module init";

  auto device_module = torch_ort_module.def_submodule("device");
  for (auto const& entry : GetORTBackendsManager().GetBackendKinds()) {
    device_module.def(
      entry.second.c_str(),
      [entry](int index) {
        return py::cast<py::object>(
          THPDevice_New(
            ORTBackendsManager::MakePyTorchDevice(
              entry.first,
              index)));
      },
      py::arg("device_index") = 0);
  }
}

} // namespace eager
} // namespace torch_ort