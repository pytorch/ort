#include <torch/extension.h>

#include "ORTUtil.h"
#include "ORTTensorImpl.h"
#include "ORTOps.h"

namespace at {
namespace native {
namespace ort {
namespace aten {

using ORTTensor = OrtValue;
using ORTTensorImpl = ORTOpaqueTensorImpl<OrtValue>;

using namespace at::native::ort::detail;

Tensor new_with_orttensor_ort(
  OrtValue&& ot,
  const TensorOptions& options) {
  const onnxruntime::Tensor& tensor = ot.Get<onnxruntime::Tensor>();
  auto &sizes = tensor.Shape().GetDims();
  //TODO: optimize it later
  auto strides = GetStride(sizes, tensor.DataType()->Size());
  return at::detail::make_tensor<ORTTensorImpl>(
    DispatchKeySet(DispatchKey::ORT),
    options.dtype(),
    at::Device(at::kORT),
    std::move(ot),
    std::vector<int64_t>(sizes.begin(), sizes.end()),
    std::vector<int64_t>(strides.begin(), strides.end()));
}

const OrtValue& orttensor_from_ort(const Tensor& tensor) {
  // FIXME: assert tensor is from ORT
  auto impl = static_cast<ORTTensorImpl*>(tensor.unsafeGetTensorImpl());
  return impl->unsafe_opaque_handle();
}

OrtValue& orttensor_from_ort(Tensor& tensor) {
  // FIXME: assert tensor is from ORT
  auto impl = static_cast<ORTTensorImpl*>(tensor.unsafeGetTensorImpl());
  return impl->unsafe_opaque_handle();
}

Tensor empty_override(
  IntArrayRef size,
  const TensorOptions& options,
  c10::optional<c10::MemoryFormat> memory_format) {
  // TODO: validate options and memory format

  ORT_LOG << "torch.empty";
  ORT_LOG << "Warning: hardcode to float type now";
  // TODO: figure out how to get the correct element type.
  OrtValue ot;
  CreateMLValue<float>(GetORTInvoker().GetCurrentExecutionProvider().GetAllocator(0, OrtMemTypeDefault),
                       size.vec(), {}, &ot);
  
  return new_with_orttensor_ort(
    std::move(ot),
    at::device(at::kORT).dtype(options.dtype()));
}

Tensor reshape(at::Tensor const& self, IntArrayRef shape) {
  ORT_LOG << "torch.reshape";

  return new_with_orttensor_ort(
    ort::detail::reshape_copy(
      orttensor_from_ort(self),
      shape.vec()),
    self.options());
}

Tensor view(const Tensor& self, IntArrayRef size) {
  ORT_LOG << "torch.view";

  return new_with_orttensor_ort(
    ort::detail::reshape_copy(
      orttensor_from_ort(self),
      at::infer_size(
        size,
        self.numel())),
    self.options());
}

TORCH_LIBRARY_IMPL(aten, ORT, m) {
  ORT_LOG << "ATen init";

  m.impl_UNBOXED("empty.memory_format", empty_override);
  m.impl("reshape", TORCH_FN(reshape));
  m.impl("view", TORCH_FN(view));
}

} // namespace aten
} // namespace ort
} // namespace native
} // namespace at