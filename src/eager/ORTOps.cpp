#include <ATen/InferSize.h>

#include <torch/extension.h>
#include "ORTOps.h"
#include "ORTUtil.h"

namespace at {
namespace native {
namespace ort {
namespace detail {

OrtValue reshape_copy(
  const OrtValue& input,
  std::vector<int64_t> shape) {
  
  // TODO: actual reshape on buffer
  const onnxruntime::Tensor& input_tensor = input.Get<onnxruntime::Tensor>();
  auto new_shape = infer_size(shape, input_tensor.Shape().Size());
  OrtValue shape_tensor;
  //todo: avoid the copy on this small shape vector;
  CreateMLValue<int64_t>(GetORTInvoker().GetCurrentExecutionProvider().GetAllocator(0, OrtMemTypeDefault),
                       {new_shape.size(),}, new_shape, &shape_tensor);
  std::vector<OrtValue> result(1);
  ORT_LOG << "Invoke ORT reshape kernel";
  auto status = GetORTInvoker().Invoke("Reshape", {input, shape_tensor}, result, nullptr);
  if (!status.IsOK())
    throw std::runtime_error("ORT return failed status: " + status.ErrorMessage());
  ORT_LOG << "Invoke ORT reshape kernel successfully";
  return result[0];
}

} // namespace detail
} // namespace ort
} // namespace native
} // namespace at
