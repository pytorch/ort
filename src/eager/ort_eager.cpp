#include <torch/extension.h>

#include "ort_backends.h"
#include "ort_util.h"

namespace torch_ort {
namespace eager {

PYBIND11_MODULE(torch_ort, m) {
  ORT_LOG << "pybind11 module init";
  m.def("get_ort_device", 
        [](const std::string& devkind, int index){
          auto& ort_backends = GetORTBackends();
          ORTBackendsManager::ORTDeviceKind ort_dev_kind;
          if (devkind == "CPU")
            ort_dev_kind = ORTBackendsManager::ORTDeviceKind::kCPU;
          else if (devkind == "Apollo")
            ort_dev_kind = ORTBackendsManager::ORTDeviceKind::kApollo;
          else
            throw std::runtime_error("Not supported device");
          return ort_backends.GetPytorchDeviceIndex(ort_dev_kind, index);
        });
}

} // namespace eager
} // namespace torch_ort