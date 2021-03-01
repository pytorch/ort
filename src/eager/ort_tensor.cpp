// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ort_tensor.h"
#include "ort_util.h"

namespace torch_ort {
namespace eager {

c10::intrusive_ptr<c10::TensorImpl> ORTTensorImpl::shallow_copy_and_detach(
  const c10::VariableVersion& version_counter,
  bool allow_tensor_metadata_change) const {
  auto impl = c10::make_intrusive<ORTTensorImpl>(
    tensor_,
    at::TensorOptions()
      .dtype(this->dtype())
      .device(this->device()));

  copy_tensor_metadata(
    this,
    impl.get(),
    version_counter,
    allow_tensor_metadata_change);

  return impl;
}

void ORTTensorImpl::shallow_copy_from(
  const c10::intrusive_ptr<TensorImpl>& impl) {
  auto* src_impl = dynamic_cast<ORTTensorImpl*>(impl.get());
  copy_tensor_metadata(
    src_impl,
    this,
    version_counter(),
    allow_tensor_metadata_change());
}

at::IntArrayRef ORTTensorImpl::sizes() const {
  const_cast<ORTTensorImpl*>(this)->cacheSizeMetadata();
  return c10::TensorImpl::sizes();
}

int64_t ORTTensorImpl::dim() const {
  const_cast<ORTTensorImpl*>(this)->cacheSizeMetadata();
  return c10::TensorImpl::dim();
}

int64_t ORTTensorImpl::numel() const {
  const_cast<ORTTensorImpl*>(this)->cacheSizeMetadata();
  return c10::TensorImpl::numel();
}

bool ORTTensorImpl::is_contiguous(at::MemoryFormat memory_format) const {
  return true;
}

int64_t ORTTensorImpl::size(int64_t d) const {
  const_cast<ORTTensorImpl*>(this)->cacheSizeMetadata();
  return c10::TensorImpl::size(d);
}

void ORTTensorImpl::cacheSizeMetadata() {
  // TODO: wrap with change generation guard
  auto& tensor = tensor_.Get<onnxruntime::Tensor>();
  auto shape = tensor.Shape();
  numel_ = shape.Size();
  sizes_ = shape.GetDims();
  strides_ = GetStrides(
    shape.GetDims(),
    tensor.DataType()->Size());
}

static const at::Storage no_storage_ = {};
const at::Storage& ORTTensorImpl::storage() const { return no_storage_; }
bool ORTTensorImpl::has_storage() const { return false; }

} // namespace eager
} // namespace torch_ort