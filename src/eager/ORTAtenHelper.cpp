#include "ORTAtenHelper.h"
#include "ORTTensorImpl.h"
#include "ORTUtil.h"

namespace torch_ort {
namespace eager {

at::Tensor new_with_orttensor_ort(
  OrtValue&& ot,
  const at::TensorOptions& options) {
  return at::Tensor(c10::make_intrusive<ORTTensorImpl>(
    std::move(ot),
    options));
}

const OrtValue& orttensor_from_ort(const at::Tensor& tensor) {
  // FIXME: assert tensor is from ORT
  auto impl = static_cast<ORTTensorImpl*>(tensor.unsafeGetTensorImpl());
  return impl->tensor();
}

OrtValue& orttensor_from_ort(at::Tensor& tensor) {
  // FIXME: assert tensor is from ORT
  auto impl = static_cast<ORTTensorImpl*>(tensor.unsafeGetTensorImpl());
  return impl->tensor();
}

} // namespace eager
} // namespace torch_ort