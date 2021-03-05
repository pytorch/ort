// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ort_aten.h"
#include "ort_tensor.h"

namespace torch_ort::eager {

#pragma region Helpers

namespace {
  inline bool is_device_supported(at::DeviceType type) {
    return type == at::kORT || type == at::kCPU;
  }

  inline void assert_tensor_supported(const at::Tensor& tensor) {
    if (tensor.is_sparse()) {
      throw std::runtime_error("ORT copy: sparse not supported");
    }

    if (tensor.is_quantized()) {
      throw std::runtime_error("ORT copy: quantized not supported");
    }

    if (!is_device_supported(tensor.device().type())) {
      throw std::runtime_error("ORT copy: device not supported");
    }
  }
}

const at::Tensor get_at_tensor_from_ort_tensor(
  OrtValue&& ot,
  const at::TensorOptions& options) {
  return at::Tensor(c10::make_intrusive<ORTTensorImpl>(
    std::move(ot),
    options));
}

const OrtValue get_ort_tensor_from_at_tensor(const at::Tensor& tensor) {
  assert_tensor_supported(tensor);

  auto* impl = dynamic_cast<ORTTensorImpl*>(tensor.unsafeGetTensorImpl());
  if (impl) {
    return impl->tensor();
  }

  OrtValue ort_tensor;
  CreateMLValue(
    tensor.data_ptr(),
    get_ort_scalar_type_from_aten(tensor.scalar_type()),
    tensor.sizes().vec(),
    &ort_tensor);
  return ort_tensor;
}

const onnxruntime::MLDataType get_ort_scalar_type_from_aten(
  at::ScalarType dtype) {
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

#pragma region Hand-Implemented ATen Ops

at::Tensor aten_empty_memory_format(
  at::IntArrayRef size,
  // *
  const at::TensorOptions& options, 
  c10::optional<at::MemoryFormat> memory_format) {
  ORT_LOG_FN(size, options, memory_format);

  // TODO: validate options and memory format
  // TODO: figure out how to get the correct element type.
  OrtValue ot;
  auto& invoker = GetORTInvoker(options.device());
  CreateMLValue(
    invoker.GetCurrentExecutionProvider().GetAllocator(0, OrtMemTypeDefault),
    get_ort_scalar_type_from_aten(at::kFloat),
    size.vec(),
    &ot);

  return get_at_tensor_from_ort_tensor(
    std::move(ot),
    options);
}

at::Tensor aten_empty_strided(
  at::IntArrayRef size,
  at::IntArrayRef stride,
  // *
  c10::optional<at::ScalarType> dtype_opt,
  c10::optional<at::Layout> layout_opt,
  c10::optional<at::Device> device_opt,
  c10::optional<bool> pin_memory_opt) {
  ORT_LOG_FN(stride, dtype_opt, layout_opt, device_opt, pin_memory_opt);

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
  return get_at_tensor_from_ort_tensor(
    std::move(ot),
    at::device(*device_opt).dtype(dtype));
}

at::Tensor aten_reshape(at::Tensor const& self, at::IntArrayRef shape) {
  ORT_LOG_FN(self, shape);

  return get_at_tensor_from_ort_tensor(
    reshape_copy(
      GetORTInvoker(self.device()),
      get_ort_tensor_from_at_tensor(self),
      shape.vec()),
    self.options());
}

at::Tensor aten_view(const at::Tensor& self, at::IntArrayRef size) {
  ORT_LOG_FN(self, size);

  return get_at_tensor_from_ort_tensor(
    reshape_copy(
      GetORTInvoker(self.device()),
      get_ort_tensor_from_at_tensor(self),
      at::infer_size(
        size,
        self.numel())),
    self.options());
}

at::Tensor& aten_copy_(
  at::Tensor& self,
  const at::Tensor& src,
  bool non_blocking) {
  ORT_LOG_FN(self, src, non_blocking);

  assert_tensor_supported(self);
  assert_tensor_supported(src);

  auto& invoker = GetORTInvoker(self.device().type() == at::kORT
    ? self.device()
    : src.device());
  const auto ort_src = get_ort_tensor_from_at_tensor(src);
  auto ort_self = get_ort_tensor_from_at_tensor(self);

  copy(invoker, ort_src, ort_self);

  return self;
}

#pragma endregion

} // namespace torch_ort::eager