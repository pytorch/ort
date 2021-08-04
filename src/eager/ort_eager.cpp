// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <torch/extension.h>
#include "torch/csrc/autograd/python_variable.h"
#include "ort_backends.h"
#include "ort_log.h"
#include "ort_aten.h"
#include "orttraining/core/framework/ortmodule_graph_builder.h"
#include "python/onnxruntime_pybind_state_common.h"
#include "core/dlpack/dlpack_python.h"
#include "ort_customops.h"

namespace onnxruntime{
namespace python{
  onnxruntime::Environment& GetEnv();
  void addGlobalMethods(py::module& m, Environment& env);
  void addObjectMethods(py::module& m, Environment& env);
  void addOrtValueMethods(pybind11::module& m);
  void addIoBindingMethods(pybind11::module& m);
  void addObjectMethodsForTraining(py::module& m);
}
}

namespace torch_ort {
namespace eager {
using namespace onnxruntime::training;

py::object ORTTensor_toDLPack(const at::Tensor& data)
{
  OrtValue ort_value = create_ort_value(data);
  return py::reinterpret_steal<py::object>(onnxruntime::dlpack::ToDlpack(ort_value));
}

at::Tensor ORTTensor_FromDLPack(const py::object& dlpack_tensor)
{
  OrtValue ort_value = onnxruntime::dlpack::FromDlpack(dlpack_tensor.ptr(), false);
  return aten_tensor_from_ort(
    std::move(ort_value),
    at::TensorOptions()
      .device(at::Device(at::DeviceType::ORT, 0)));
}

PYBIND11_MODULE(torch_ort, torch_ort_module) {
  ORT_LOG_DEBUG << "pybind11 module init";

  torch_ort_module.def(
    "device",
    [](int device_index) {
      return py::cast<py::object>(
        THPDevice_New(at::Device(at::DeviceType::ORT, device_index)));
    },
    py::arg("device_index") = 0);

  onnxruntime::Environment& env = onnxruntime::python::GetEnv();

  onnxruntime::python::addGlobalMethods(torch_ort_module, env);
  onnxruntime::python::addObjectMethods(torch_ort_module, env);
  onnxruntime::python::addOrtValueMethods(torch_ort_module);
  onnxruntime::python::addIoBindingMethods(torch_ort_module);
  onnxruntime::python::addObjectMethodsForTraining(torch_ort_module);
  
  torch_ort_module.def("ort_to_dlpack", [](at::Tensor data) {
    return ORTTensor_toDLPack(data);
  });
  torch_ort_module.def("ort_from_dlpack", [](py::object dlpack_tensor) {
    return ORTTensor_FromDLPack(dlpack_tensor);
  });

  auto customop_module = torch_ort_module.def_submodule("custom_ops");
  torch_ort::eager::GenerateCustomOpsBindings(customop_module);
}

} // namespace eager
} // namespace torch_ort