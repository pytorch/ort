#pragma once

#include <ATen/OpaqueTensorImpl.h>
#include <ATen/WrapDimUtils.h>

namespace at {

template <typename OpaqueHandle>
struct ORTOpaqueTensorImpl : public OpaqueTensorImpl<OpaqueHandle> {
  ORTOpaqueTensorImpl(
    at::DispatchKeySet key_set,
    const caffe2::TypeMeta& data_type,
    c10::Device device,
    OpaqueHandle opaque_handle,
    c10::IntArrayRef sizes,
    c10::IntArrayRef strides)
    : OpaqueTensorImpl<OpaqueHandle>(
      key_set,
      data_type,
      device,
      opaque_handle,
      sizes),
      strides_(strides.vec()) {  
  }

  IntArrayRef strides() const override {
    return strides_;
  }

  bool is_contiguous(
    c10::MemoryFormat memory_format
      = c10::MemoryFormat::Contiguous) const override {
    return true;
  }

  int64_t stride(int64_t d) const override {
    d = at::maybe_wrap_dim(d, this->dim(), false);
    return strides_[d];
  }

private:
  SmallVector<int64_t, 5> strides_;
};

} // namespace at