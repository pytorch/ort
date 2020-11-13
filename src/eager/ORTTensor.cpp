#include "ORTTensor.h"

namespace at {
namespace native {
namespace ort {
namespace detail {

class ORTTensor::Impl final {
public:
  explicit Impl(std::vector<int64_t> sizes)
    : sizes_(std::move(sizes)),
      strides_(std::vector<int64_t>(sizes_.size())),
      numel_(std::accumulate(
        std::begin(sizes_),
        std::end(sizes_),
        1,
        std::multiplies<int64_t>())) {
  }

  std::vector<int64_t> sizes() const {
    return sizes_;
  }

  std::vector<int64_t> strides() const {
    return strides_;
  }

  inline int64_t dim() const {
    return sizes_.size();
  }

  inline int64_t numel() const {
    return numel_;
  }

private:
  std::vector<int64_t> sizes_;
  std::vector<int64_t> strides_;
  int64_t numel_;
};

ORTTensor::ORTTensor(std::vector<int64_t> sizes)
  : impl_(std::make_shared<Impl>(std::move(sizes))) {
}

std::shared_ptr<ORTTensor::Impl> ORTTensor::impl() {
  return impl_;
}

std::shared_ptr<const ORTTensor::Impl> ORTTensor::impl() const {
  return impl_;
}

std::vector<int64_t> ORTTensor::sizes() const {
  return impl()->sizes();
}

std::vector<int64_t> ORTTensor::strides() const {
  return impl()->strides();
}

int64_t ORTTensor::dim() const {
  return impl()->dim();
}

int64_t ORTTensor::numel() const {
  return impl()->numel();
}

} // namespace detail
} // namespace ort
} // namespace native
} // namespace at