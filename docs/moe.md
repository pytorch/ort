# Introduction

Mixture of Experts technique, frequently referred as MoE, to scale Transformer models 
without incurring high computational resource requirements is gaining traction. There are 
four major subcomponents used to deploy Mixture of Experts technique.

1.	Experts are at the heart of Mixture of Experts technique. Usually, standard feedforward 
neural network sublayer is used as an expert but that is not necessary. Two Expert 
implementations are available.
- <code>FFNExpert</code> is a standard feedforward neural network.
- <code>MergedFFNExpert</code> is optimized implementation of feeedforward neural network.
It takes advantage of batched matrix-multiplication kernels provided by CUDA.
 
2.	Fundamentally, the Mixture of Experts technique deploys many experts in the model to 
scale weights count to achieve better model quality. However, at runtime only small subset 
of these experts are used to process the given input tokens. This allows data scientists 
to keep FLOP budget under control while increasing model size. A gating function is used 
to select small subset of expert at runtime. Top1 or Top2 algorithms are popular choices 
as gating functions. <code> Top1Gating </code> and <code> Top2Gating </code> functions are
implemented.
 
3.	MixtureOfExperts sublayer consists of gating function, experts and needed communication
collectives to synchronize experts across multiple shards. Two implementation of MixtureOfExperts
sublayer are available.
- <code>MixtureOfExpertsEP</code> enables seamless Expert Parallelism for distribued training.
- <code>MixtureOfExpertsES</code> aims to implement Megatron style sharding of experts.
 
4.	Finally, a Transformer layer (encoder or decoder) is constructed using a multi-headed
attention layer, gating function, experts and MixtureOfExperts sub layer. This entire layer
is referred as MoELayer.

- <code>TransformerMoEEncoder </code> implements standard transformer encoder using <code>MixtureOfExpertsEP</code>.
- <code>TransformerMoEDecoder </code> implements standard transformer decoder using <code>MixtureOfExpertsEP</code>.
- <code>LanguageExpertMoEEncoder </code> implements language transformer encoder using <code>MixtureOfExpertsEP</code>.
- <code>LanguageExpertMoEDecoder </code> implements language transformer encoder using <code>MixtureOfExpertsEP</code>.


# TODO Lost functions

# TODO Utilities
