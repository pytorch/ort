#include <torch/extension.h>
#include <ATen/Context.h>
#include <torch/library.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/Tensor.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorImpl.h>

using namespace at;

class ORTTensorImpl : public c10::TensorImpl {
 public:
  explicit ORTTensorImpl(
    c10::optional<ScalarType> dtype,
    c10::optional<Device> device);
};

ORTTensorImpl::ORTTensorImpl(
  c10::optional<ScalarType> dtype,
  c10::optional<Device> device) : c10::TensorImpl(
    c10::DispatchKeySet {
      c10::DispatchKey::ORT,
      c10::DispatchKey::AutogradORT
    },
    c10::scalarTypeToTypeMeta(dtype.value()),
    device)
{
}

// aten::zeros(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
Tensor ort_zeros(
  IntArrayRef size,
  c10::optional<ScalarType> dtype,
  c10::optional<Layout> layout,
  c10::optional<Device> device,
  c10::optional<bool> pin_memory) {
  Tensor r;
  return r;
}

TORCH_LIBRARY_IMPL(aten, ORT, m) {
  m.impl("zeros", ort_zeros);
}

PYBIND11_MODULE(torch_ort, m) {
//TODO: init ort backend
}
