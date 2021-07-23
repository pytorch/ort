#include <torch/extension.h>
#include <vector>

namespace torch_ort {
namespace eager {
namespace msnpu {

static const char * const TransformerDecoderName = "transformerdecoder";

std::vector<at::Tensor> transformerdecoder(
    int64_t padded_hidden_size, int64_t head_size, float soft_dropout_prob,
    int64_t soft_dropout_seed, float dense_dropout_prob,
    int64_t dense_dropout_seed, float mlp_dropout_prob,
    int64_t mlp_dropout_seed, float epsilon,
    const torch::Tensor& embeddings_post_dropout,
    const torch::Tensor& normalization_1_w,
    const torch::Tensor& normalization_1_b, const torch::Tensor &query_w,
    const torch::Tensor& query_b, const torch::Tensor &key_w,
    const torch::Tensor& key_b, const torch::Tensor &value_w,
    const torch::Tensor& value_b, const torch::Tensor &attention_mask,
    const torch::Tensor& project_w, const torch::Tensor &project_b,
    const torch::Tensor& FFN1_w, const torch::Tensor &FFN1_b,
    const torch::Tensor& FFN2_w, const torch::Tensor &FFN2_b,
    const torch::Tensor& normalization_2_w,
    const torch::Tensor& normalization_2_b, const torch::Tensor &pad_values);
}
} // namespace eager
} // namespace torch_ort