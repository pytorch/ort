#include <torch/extension.h>

#include "ORTUtil.h"

#include <memory>


namespace at {
namespace native {
namespace ort {
namespace detail {

PYBIND11_MODULE(torch_ort, m) {
  ORT_LOG << "pybind11 module init";
}

} // namespace detail
} // namespace ort
} // namespace native
} // namespace at
