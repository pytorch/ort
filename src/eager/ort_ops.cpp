// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ort_ops.h"
#include "ort_util.h"

namespace torch_ort {
namespace eager {

OrtValue reshape_copy(
  onnxruntime::ORTInvoker& invoker,
  const OrtValue& input,
  std::vector<int64_t> shape) {
  
  // TODO: actual reshape on buffer
  const onnxruntime::Tensor& input_tensor = input.Get<onnxruntime::Tensor>();
  auto new_shape = at::infer_size(shape, input_tensor.Shape().Size());
  OrtValue shape_tensor;
  //todo: avoid the copy on this small shape vector;
  auto element_type = onnxruntime::DataTypeImpl::GetType<int64_t>();
  CreateMLValue(invoker.GetCurrentExecutionProvider().GetAllocator(0, OrtMemTypeDefault),
                element_type, {(int64_t)new_shape.size(),}, &shape_tensor);
  auto* ort_shape_tensor = shape_tensor.GetMutable<onnxruntime::Tensor>();
  CopyVectorToTensor<int64_t>(new_shape, *ort_shape_tensor);
  std::vector<OrtValue> result(1);
  ORT_LOG << "Invoke ORT reshape kernel";
  auto status = invoker.Invoke("Reshape", {input, shape_tensor}, result, nullptr);
  if (!status.IsOK())
    throw std::runtime_error("ORT return failure status: " + status.ErrorMessage());
  ORT_LOG << "Invoke ORT reshape kernel successfully";
  return result[0];
}

OrtValue add(onnxruntime::ORTInvoker& invoker,
             const OrtValue& A,
             const OrtValue& B){
  std::vector<OrtValue> result(1);
  ORT_LOG << "Invoke ORT Add kernel";
  auto status = invoker.Invoke("Add", {A, B}, result, nullptr);
  if (!status.IsOK())
    throw std::runtime_error("ORT return failure status: " + status.ErrorMessage());
  ORT_LOG << "Invoke ORT Add kernel successfully";
  return result[0];
}

void copy(onnxruntime::ORTInvoker& invoker, 
          const OrtValue& src, OrtValue& dst){
  ORT_LOG << "Invoke ORT Copy ";
  auto& ort_ep = invoker.GetCurrentExecutionProvider();
  
  const auto& src_tensor = src.Get<onnxruntime::Tensor>();
  auto* dst_tensor = dst.GetMutable<onnxruntime::Tensor>();
  if (!dst_tensor)
    throw std::runtime_error("ORT copy: dst is not a tensor");
  ort_ep.GetDataTransfer()->CopyTensor(src_tensor, *dst_tensor);
}

} // namespace eager
} // namespace torch_ort