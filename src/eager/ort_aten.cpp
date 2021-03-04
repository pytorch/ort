// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ort_aten.h"
#include "ort_tensor.h"

namespace torch_ort {
namespace eager {

#pragma region Helpers

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

onnxruntime::MLDataType get_ort_scalar_type_from_aten(at::ScalarType dtype){
  switch (dtype){
    case at::kFloat:
      return onnxruntime::DataTypeImpl::GetType<float>();
    case at::kDouble:
      return onnxruntime::DataTypeImpl::GetType<double>();
    case at::kHalf:
      return onnxruntime::DataTypeImpl::GetType<onnxruntime::MLFloat16>();
    case at::kBFloat16:
      return onnxruntime::DataTypeImpl::GetType<onnxruntime::BFloat16>();
    case at::kInt:
      return onnxruntime::DataTypeImpl::GetType<int>();
    case at::kShort:
      return onnxruntime::DataTypeImpl::GetType<int16_t>();
    case at::kLong:
      return onnxruntime::DataTypeImpl::GetType<int64_t>();
    default:
      ORT_THROW("Unsupport aten scalar type: ", dtype);
  }
}

#pragma endregion

at::Tensor ort_aten_empty_memory_format(
  at::IntArrayRef size,
  // *
  const at::TensorOptions& options, 
  c10::optional<at::MemoryFormat> memory_format) {
  // TODO: validate options and memory format

  ORT_LOG << "torch.empty";

  ORT_LOG << "Warning: hardcode to float type now";
  ORT_LOG << "Device is: " << options.device();
  ORT_LOG << "Device id is: " << options.device().index();
  // TODO: figure out how to get the correct element type.
  OrtValue ot;
  auto& invoker = GetORTInvoker(options.device());
  CreateMLValue(
    invoker.GetCurrentExecutionProvider().GetAllocator(0, OrtMemTypeDefault),
    get_ort_scalar_type_from_aten(at::kFloat),
    size.vec(),
    &ot);

  return new_with_orttensor_ort(
    std::move(ot),
    options);
}

at::Tensor ort_aten_empty_strided(
  at::IntArrayRef size,
  at::IntArrayRef stride,
  // *
  c10::optional<at::ScalarType> dtype_opt,
  c10::optional<at::Layout> layout_opt,
  c10::optional<at::Device> device_opt,
  c10::optional<bool> pin_memory_opt) {
  ORT_LOG << "aten::empty_strided";
  // TODO: handle stride
  // TODO: how to handle type conversion
  OrtValue ot;
  assert(device_opt.has_value());
  // TODO: how to support layout
  assert(!layout_opt.has_value());
  at::ScalarType dtype = c10::dtype_or_default(dtype_opt);
  auto& invoker = GetORTInvoker(*device_opt);
  CreateMLValue(
    invoker.GetCurrentExecutionProvider().GetAllocator(0, OrtMemTypeDefault),
    get_ort_scalar_type_from_aten(dtype),
    size.vec(),
    &ot);
  return new_with_orttensor_ort(
    std::move(ot),
    at::device(*device_opt).dtype(dtype));
}

at::Tensor ort_aten_reshape(at::Tensor const& self, at::IntArrayRef shape) {
  ORT_LOG << "torch.reshape";

  return new_with_orttensor_ort(
    reshape_copy(
      GetORTInvoker(self.device()),
      orttensor_from_ort(self),
      shape.vec()),
    self.options());
}

at::Tensor ort_aten_view(const at::Tensor& self, at::IntArrayRef size) {
  ORT_LOG << "torch.view";

  return new_with_orttensor_ort(
    reshape_copy(
      GetORTInvoker(self.device()),
      orttensor_from_ort(self),
      at::infer_size(
        size,
        self.numel())),
    self.options());
}

namespace{
  inline bool is_device_supported(at::DeviceType type){
    return type == at::kORT || type == at::kCPU;
  }

  OrtValue ort_tensor_from_at_tensor(at::Tensor& tensor){
    assert(tensor.device().type() == at:kCPU);
    tensor.data_ptr();
    //todo: figure out the correct type
    OrtValue ot;
    CreateMLValue(tensor.data_ptr(), get_ort_scalar_type_from_aten(tensor.scalar_type()), tensor.sizes().vec(), &ot);
    return ot;
  }

  OrtValue ort_tensor_from_at_tensor(const at::Tensor& tensor){
    assert(tensor.device().type() == at:kCPU);
    tensor.data_ptr();
    //todo: figure out the correct type
    OrtValue ot;
    CreateMLValue(tensor.data_ptr(), get_ort_scalar_type_from_aten(tensor.scalar_type()), tensor.sizes().vec(), &ot);
    return ot;
  }
}

at::Tensor& ort_aten_copy_(at::Tensor & self, const at::Tensor & src, bool non_blocking){
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
    copy(
      GetORTInvoker(self.device()),
      ort_tensor_from_at_tensor(src),
      orttensor_from_ort(self));
  } else if (self.device().type() == at::kCPU && src.device().type() == at::kORT){
    auto ort_self = ort_tensor_from_at_tensor(self);
    copy(
      GetORTInvoker(src.device()),
      orttensor_from_ort(src),
      ort_self);
  }

  return self;
}

} // namespace eager
} // namespace torch_ort