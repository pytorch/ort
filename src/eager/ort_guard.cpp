#include <torch/extension.h>

#include "ort_util.h"

namespace torch_ort {
namespace eager {

struct ORTGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  ORTGuardImpl() {
    ORT_LOG << "ORTGuardImpl()";
  }

  ORTGuardImpl(at::DeviceType t) {
    ORT_LOG << "ORTGuardImpl(" << t << ")";
    AT_ASSERT(t == at::DeviceType::ORT);
  }
  
  at::DeviceType type() const override {
    ORT_LOG << "ORTGuardImpl::type()";
    return at::DeviceType::ORT;
  }
  
  at::Device exchangeDevice(at::Device d) const override {
    ORT_LOG << "ORTGuardImpl::exchangeDevice(" << d << ")";
    AT_ASSERT(d.type() == at::DeviceType::ORT);
    AT_ASSERT(d.index() == 0);
    return d;
  }

  at::Device getDevice() const override {
    ORT_LOG << "ORTGuardImpl::getDevice()";
    return at::Device(at::DeviceType::ORT, 0);
  }
  
  void setDevice(at::Device d) const override {
    ORT_LOG << "ORTGuardImpl::setDevice(" << d << ")";
    AT_ASSERT(d.type() == at::DeviceType::ORT);
    AT_ASSERT(d.index() == 0);
  }
  
  void uncheckedSetDevice(at::Device d) const noexcept override {
    ORT_LOG << "ORTGuardImpl::uncheckedSetDevice(" << d << ")";
  }
  
  at::Stream getStream(at::Device d) const noexcept override {
    ORT_LOG << "ORTGuardImpl::getStream(" << d << ")";
    return at::Stream(at::Stream::DEFAULT, at::Device(at::DeviceType::ORT, 0));
  }
  
  at::Stream exchangeStream(at::Stream s) const noexcept override {
    ORT_LOG << "ORTGuardImpl::exchangeStream(" << s << ")";
    return at::Stream(at::Stream::DEFAULT, at::Device(at::DeviceType::ORT, 0));
  }
  
  at::DeviceIndex deviceCount() const noexcept override {
    ORT_LOG << "ORTGuardImpl::deviceCount()";
    return 1;
  }

  #pragma region events

  void record(void** event,
    const at::Stream& stream,
    const at::DeviceIndex device_index,
    const at::EventFlag flag) const override {
    TORCH_CHECK(false, "ORT backend doesn't support events.");
  }

  void block(
    void* event,
    const at::Stream& stream) const override {
    TORCH_CHECK(false, "ORT backend doesn't support events.");
  }
  
  bool queryEvent(void* event) const override {
    TORCH_CHECK(false, "ORT backend doesn't support events.");
  }
  
  void destroyEvent(
    void* event,
    const at::DeviceIndex device_index) const noexcept override { }

  #pragma endregion events
};

C10_REGISTER_GUARD_IMPL(ORT, ORTGuardImpl);

} // namespace eager
} // namespace torch_ort