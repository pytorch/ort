#include "ORTAtenHelper.h"
#include "ORTTensorImpl.h"
#include "ORTUtil.h"

namespace at {
namespace native {
namespace ort {
namespace aten {

using namespace at::native::ort::detail;
using ORTTensorImpl = ORTOpaqueTensorImpl<OrtValue>;

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


}
}
}
}