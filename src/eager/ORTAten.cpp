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

Tensor add(const Tensor& A, const Tensor& B, c10::Scalar alpha=1) {
  ORT_LOG << "torch.add";
  //todo: handle alpha
  return new_with_orttensor_ort(
    ort::detail::add(
      orttensor_from_ort(A),
      orttensor_from_ort(B)),
    A.options());
}

namespace{
  inline bool is_device_supported(DeviceType type){
    return type == at::kORT || type == at::kCPU;
  }

  ORTTensor ort_tensor_from_at_tensor(Tensor& tensor){
    assert(tensor.device().type() == at:kCPU);
    tensor.data_ptr();
    //todo: figure out the correct type
    OrtValue ot;
    CreateMLValue<float>(tensor.data_ptr(), tensor.sizes().vec(), &ot);
    return ot;
  }

  ORTTensor ort_tensor_from_at_tensor(const Tensor& tensor){
    assert(tensor.device().type() == at:kCPU);
    tensor.data_ptr();
    //todo: figure out the correct type
    OrtValue ot;
    CreateMLValue<float>(tensor.data_ptr(), tensor.sizes().vec(), &ot);
    return ot;
  }
}

Tensor& copy_tensor(Tensor & self, const Tensor & src, bool non_blocking){
  if (self.is_sparse() || src.is_sparse()){
    throw std::runtime_error("ORT copy: sparse not supported");
  }
  if (self.is_quantized() || src.is_quantized()){
    throw std::runtime_error("ORT copy: quantized not supported");
  }

  if (!is_device_supported(src.device().type()) || !is_device_supported(self.device().type())){
    throw std::runtime_error("ORT copy: device not supported");
  }
  //TODO: more flexible way to dispatch the copy implementation
  if (self.device().type() == at::kORT && src.device().type() == at::kCPU){
    ort::detail::copy(
      ort_tensor_from_at_tensor(src),
      orttensor_from_ort(self));
  } else if (self.device().type() == at::kCPU && src.device().type() == at::kORT){
    ORTTensor ort_self = ort_tensor_from_at_tensor(self);
    ort::detail::copy(
      orttensor_from_ort(src),
      ort_self);
  }

  return self;
}

TORCH_LIBRARY_IMPL(aten, ORT, m) {
  ORT_LOG << "ATen init";

  m.impl_UNBOXED("empty.memory_format", empty_override);
  m.impl("reshape", TORCH_FN(reshape));
  m.impl("view", TORCH_FN(view));
  m.impl("aten::add.Tensor", TORCH_FN(add));
  m.impl("copy_", TORCH_FN(copy_tensor));
}

} // namespace aten
} // namespace ort
} // namespace native
} // namespace at