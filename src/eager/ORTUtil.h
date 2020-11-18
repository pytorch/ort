#pragma once

// FIXME: no idea how to increase logging level for INFO, DEBUG, etc
#define ORT_LOG LOG(WARNING) << "[ORT] "

#include "core/framework/ml_value.h"
#include "core/framework/tensor.h"
#include "core/eager/ort_kernel_invoker.h"

namespace at {
namespace native {
namespace ort {
namespace detail {

onnxruntime::ORTInvoker& GetORTInvoker();

template <typename T>
inline void CopyVectorToTensor(const std::vector<T>& value, onnxruntime::Tensor& tensor) {
  gsl::copy(gsl::make_span(value), tensor.MutableDataAsSpan<T>());
}

// vector<bool> is specialized so we need to handle it separately
template <>
inline void CopyVectorToTensor<bool>(const std::vector<bool>& value, onnxruntime::Tensor& tensor) {
  auto output_span = tensor.MutableDataAsSpan<bool>();
  for (size_t i = 0, end = value.size(); i < end; ++i) {
    output_span[i] = value[i];
  }
}

template <typename T>
void CreateMLValue(onnxruntime::AllocatorPtr alloc, const std::vector<int64_t>& dims, const std::vector<T>& value,
                   OrtValue* p_mlvalue) {
  onnxruntime::TensorShape shape(dims);
  auto element_type = onnxruntime::DataTypeImpl::GetType<T>();
  std::unique_ptr<onnxruntime::Tensor> p_tensor = onnxruntime::make_unique<onnxruntime::Tensor>(element_type,
                                                                      shape,
                                                                      alloc);
  if (value.size() > 0) {
    CopyVectorToTensor(value, *p_tensor);
  }

  p_mlvalue->Init(p_tensor.release(),
                  onnxruntime::DataTypeImpl::GetType<onnxruntime::Tensor>(),
                  onnxruntime::DataTypeImpl::GetType<onnxruntime::Tensor>()->GetDeleteFunc());
}

std::vector<int64_t> GetStride(const std::vector<int64_t>& shape, int64_t element_size);

} // namespace detail
} // namespace ort
} // namespace native
} // namespace at