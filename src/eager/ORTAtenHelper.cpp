#include "ORTAtenHelper.h"
#include "ORTTensorImpl.h"
#include "ORTUtil.h"

namespace at {
namespace native {
namespace ort {
namespace aten {

using namespace at::native::ort::detail;
using ORTTensorImpl = ORTOpaqueTensorImpl<OrtValue>;

  OrtValue&& ot,
  const at::TensorOptions& options) {
  return at::Tensor(c10::make_intrusive<ORTTensorImpl>(
    std::move(ot),
    options));
}

const OrtValue& orttensor_from_ort(const Tensor& tensor) {
  // FIXME: assert tensor is from ORT
  auto impl = static_cast<ORTTensorImpl*>(tensor.unsafeGetTensorImpl());
  return impl->tensor();
}

OrtValue& orttensor_from_ort(Tensor& tensor) {
  // FIXME: assert tensor is from ORT
  auto impl = static_cast<ORTTensorImpl*>(tensor.unsafeGetTensorImpl());
  return impl->tensor();
}


}
}
}
}