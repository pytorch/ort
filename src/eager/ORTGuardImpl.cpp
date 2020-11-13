#include <torch/extension.h>

#include "ORTUtil.h"

namespace at {
namespace detail {

struct ORTGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  ORTGuardImpl() {
    ORT_LOG << "ORTGuardImpl()";
  }

  ORTGuardImpl(DeviceType t) {
    ORT_LOG << "ORTGuardImpl(" << t << ")";
    AT_ASSERT(t == DeviceType::ORT);
  }
  
  DeviceType type() const override {
    ORT_LOG << "ORTGuardImpl::type()";
    return DeviceType::ORT;
  }
  
  Device exchangeDevice(Device d) const override {
    ORT_LOG << "ORTGuardImpl::exchangeDevice(" << d << ")";
    AT_ASSERT(d.type() == DeviceType::ORT);
    AT_ASSERT(d.index() == 0);
    return d;
  }

  Device getDevice() const override {
    ORT_LOG << "ORTGuardImpl::getDevice()";
    return Device(DeviceType::ORT, 0);
  }
  
  void setDevice(Device d) const override {
    ORT_LOG << "ORTGuardImpl::setDevice(" << d << ")";
    AT_ASSERT(d.type() == DeviceType::ORT);
    AT_ASSERT(d.index() == 0);
  }
  
  void uncheckedSetDevice(Device d) const noexcept override {
    ORT_LOG << "ORTGuardImpl::uncheckedSetDevice(" << d << ")";
  }
  
  Stream getStream(Device d) const noexcept override {
    ORT_LOG << "ORTGuardImpl::getStream(" << d << ")";
    return Stream(Stream::DEFAULT, Device(DeviceType::ORT, 0));
  }
  
  Stream exchangeStream(Stream s) const noexcept override {
    ORT_LOG << "ORTGuardImpl::exchangeStream(" << s << ")";
    return Stream(Stream::DEFAULT, Device(DeviceType::ORT, 0));
  }
  
  DeviceIndex deviceCount() const noexcept override {
    ORT_LOG << "ORTGuardImpl::deviceCount()";
    return 1;
  }

  #pragma region events

  void record(void** event,
    const Stream& stream,
    const DeviceIndex device_index,
    const EventFlag flag) const override {
    TORCH_CHECK(false, "ORT backend doesn't support events.");
  }

  void block(
    void* event,
    const Stream& stream) const override {
    TORCH_CHECK(false, "ORT backend doesn't support events.");
  }
  
  bool queryEvent(void* event) const override {
    TORCH_CHECK(false, "ORT backend doesn't support events.");
  }
  
  void destroyEvent(
    void* event,
    const DeviceIndex device_index) const noexcept override { }

  #pragma endregion events
};

C10_REGISTER_GUARD_IMPL(ORT, ORTGuardImpl);

} // namespace detail
} // namespace at