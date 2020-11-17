#pragma once

#include <vector>
#include <numeric>
#include <cstdint>
#include <memory>

namespace at {
namespace native {
namespace ort {
namespace detail {

class ORTTensor final {
  class Impl;

public:
  ORTTensor() = default;
  explicit ORTTensor(std::vector<int64_t> sizes);
  ~ORTTensor() = default;

  ORTTensor(ORTTensor&&) = default;
  ORTTensor& operator=(ORTTensor&&) = default;

  ORTTensor(const ORTTensor&) = default;
  ORTTensor& operator=(const ORTTensor&) = default;

  bool defined() const {
    return static_cast<bool>(impl_);
  }

  std::vector<int64_t> sizes() const;
  std::vector<int64_t> strides() const;
  int64_t dim() const;
  int64_t numel() const;

private:
  std::shared_ptr<Impl> impl();
  std::shared_ptr<const Impl> impl() const;
  std::shared_ptr<Impl> impl_;
};

} // namespace detail
} // namespace ort
} // namespace native
} // namespace at
