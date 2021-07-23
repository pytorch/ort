#include "msnpu_ops.h"
#include "ort_aten.h"
#include "ort_backends.h"

namespace torch_ort {
namespace eager {
namespace msnpu {

std::vector<at::Tensor> transformerdecoder(
    int64_t padded_hidden_size, int64_t head_size, float soft_dropout_prob,
    int64_t soft_dropout_seed, float dense_dropout_prob,
    int64_t dense_dropout_seed, float mlp_dropout_prob,
    int64_t mlp_dropout_seed, float epsilon,
    const torch::Tensor& embeddings_post_dropout,
    const torch::Tensor& normalization_1_w,
    const torch::Tensor& normalization_1_b, const torch::Tensor& query_w,
    const torch::Tensor& query_b, const torch::Tensor& key_w,
    const torch::Tensor& key_b, const torch::Tensor& value_w,
    const torch::Tensor& value_b, const torch::Tensor& attention_mask,
    const torch::Tensor& project_w, const torch::Tensor& project_b,
    const torch::Tensor& FFN1_w, const torch::Tensor& FFN1_b,
    const torch::Tensor& FFN2_w, const torch::Tensor& FFN2_b,
    const torch::Tensor& normalization_2_w,
    const torch::Tensor& normalization_2_b, const torch::Tensor& pad_values) {

  auto& invoker = GetORTInvoker(embeddings_post_dropout.device());
  constexpr size_t num_outputs = 18;
  constexpr size_t num_attrs = 8;
  const  std::string ort_op_name = "TransformerDecoder";

  // Create ORT attributes
  onnxruntime::NodeAttributes attrs(num_attrs);
  attrs["paddedHiddenSize"] = create_ort_attribute(
      "paddedHiddenSize", padded_hidden_size, at::ScalarType::Long);
  attrs["headSize"] =
      create_ort_attribute("headSize", head_size, at::ScalarType::Long);
  attrs["softDropoutProb"] = create_ort_attribute(
      "softDropoutProb", soft_dropout_prob, at::ScalarType::Float);
  attrs["softDropoutSeed"] = create_ort_attribute(
      "softDropoutSeed", soft_dropout_seed, at::ScalarType::Long);
  attrs["denseDropoutProb"] = create_ort_attribute(
      "denseDropoutProb", dense_dropout_prob, at::ScalarType::Float);
  attrs["denseDropoutSeed"] = create_ort_attribute(
      "denseDropoutSeed", dense_dropout_seed, at::ScalarType::Long);
  attrs["denseDropoutProb"] = create_ort_attribute(
      "denseDropoutProb", mlp_dropout_prob, at::ScalarType::Float);
  attrs["mlpDropoutProb"] = create_ort_attribute(
      "mlpDropoutProb", dense_dropout_seed, at::ScalarType::Float);
  attrs["mlpDropoutSeed"] = create_ort_attribute(
      "mlpDropoutSeed", mlp_dropout_seed, at::ScalarType::Long);
  attrs["epsilon"] =
      create_ort_attribute("epsilon", epsilon, at::ScalarType::Float);

  // Create ORTValues for input tensors
  auto ort_in_embeddings_post_dropout =
      create_ort_value(invoker, embeddings_post_dropout);
  auto ort_in_normalization_1_w = create_ort_value(invoker, normalization_1_w);
  auto ort_in_normalization_1_b = create_ort_value(invoker, normalization_1_b);
  auto ort_in_query_w = create_ort_value(invoker, query_w);
  auto ort_in_query_b = create_ort_value(invoker, query_b);
  auto ort_in_key_w = create_ort_value(invoker, key_w);
  auto ort_in_key_b = create_ort_value(invoker, key_b);
  auto ort_in_value_w = create_ort_value(invoker, value_w);
  auto ort_in_value_b = create_ort_value(invoker, value_b);
  auto ort_in_attention_mask = create_ort_value(invoker, attention_mask);
  auto ort_in_project_w = create_ort_value(invoker, project_w);
  auto ort_in_project_b = create_ort_value(invoker, project_b);
  auto ort_in_FFN1_w = create_ort_value(invoker, FFN1_w);
  auto ort_in_FFN1_b = create_ort_value(invoker, FFN1_b);
  auto ort_in_FFN2_w = create_ort_value(invoker, FFN2_w);
  auto ort_in_FFN2_b = create_ort_value(invoker, FFN2_b);
  auto ort_in_normalization_2_w = create_ort_value(invoker, normalization_2_w);
  auto ort_in_normalization_2_b = create_ort_value(invoker, normalization_2_b);
  auto ort_in_pad_values = create_ort_value(invoker, pad_values);

  // Create ORTValues for output tensors.
  std::vector<OrtValue> ort_outputs(num_outputs);

  // Invoke the transformer decoder command on the ORT device
  auto status = invoker.Invoke(
      ort_op_name,
      {ort_in_embeddings_post_dropout, ort_in_normalization_1_w,
       ort_in_normalization_1_b, ort_in_query_w, ort_in_query_b, ort_in_key_w,
       ort_in_key_b, ort_in_value_w, ort_in_value_b, ort_in_attention_mask,
       ort_in_project_w, ort_in_project_b, ort_in_FFN1_w, ort_in_FFN1_b,
       ort_in_FFN2_w, ort_in_FFN2_b, ort_in_normalization_2_w,
       ort_in_normalization_2_b, ort_in_pad_values},
      ort_outputs, &attrs, onnxruntime::kMSDomain);

  if (!status.IsOK()) {
    throw std::runtime_error("ORT returned a failure status: " +
                             status.ErrorMessage());
  }

  // Transform outputs into torch tensors
  std::vector<at::Tensor> outputs(num_outputs);
  for (size_t i = 0; i < num_outputs; i++) {
    outputs[i] = aten_tensor_from_ort(std::move(ort_outputs[i]),
                                      embeddings_post_dropout.options());
  }

  return outputs;
}
} // namespace msnpu
} // namespace eager
} // namespace torch_ort
