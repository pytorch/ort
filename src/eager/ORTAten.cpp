#include <torch/extension.h>

#include "ORTUtil.h"
#include "ORTTensorImpl.h"
#include "ORTOps.h"
#include "ORTAtenHelper.h"


namespace at {
namespace native {
namespace ort {
namespace aten {

using ORTTensor = OrtValue;
using ORTTensorImpl = ORTOpaqueTensorImpl<OrtValue>;

using namespace at::native::ort::detail;

Tensor ort_aten_empty_memory_format(IntArrayRef size, 
  const TensorOptions& options, 
  c10::optional<MemoryFormat> memory_format) {
  // TODO: validate options and memory format

  ORT_LOG << "torch.empty";
  ORT_LOG << "Warning: hardcode to float type now";
  ORT_LOG << "Device is: " << options.device();
  ORT_LOG << "Device id is: " << options.device().index();
  // TODO: figure out how to get the correct element type.
  OrtValue ot;
  auto& invoker = GetORTInvoker(options.device());
  CreateMLValue<float>(invoker.GetCurrentExecutionProvider().GetAllocator(0, OrtMemTypeDefault),
                       size.vec(), {}, &ot);
  
  return new_with_orttensor_ort(
    std::move(ot),
    at::device(at::kORT).dtype(options.dtype()));
}

Tensor ort_aten_reshape(at::Tensor const& self, IntArrayRef shape) {
  ORT_LOG << "torch.reshape";

  return new_with_orttensor_ort(
    ort::detail::reshape_copy(
      GetORTInvoker(self.device()),
      orttensor_from_ort(self),
      shape.vec()),
    self.options());
}

Tensor ort_aten_view(const Tensor& self, IntArrayRef size) {
  ORT_LOG << "torch.view";

  return new_with_orttensor_ort(
    ort::detail::reshape_copy(
      GetORTInvoker(self.device()),
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
      GetORTInvoker(A.device()),
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

Tensor& ort_aten_copy_(Tensor & self, const Tensor & src, bool non_blocking){
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
      GetORTInvoker(self.device()),
      ort_tensor_from_at_tensor(src),
      orttensor_from_ort(self));
  } else if (self.device().type() == at::kCPU && src.device().type() == at::kORT){
    ORTTensor ort_self = ort_tensor_from_at_tensor(self);
    ort::detail::copy(
      GetORTInvoker(src.device()),
      orttensor_from_ort(src),
      ort_self);
  }

  return self;
}

} // namespace aten
} // namespace ort
} // namespace native
} // namespace at