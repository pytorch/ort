# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""
  ORT MoE Configuration Options
  -----------------------------
  {
    ## Enable Verbose output
    "verbose" : False|True,

    ## Options to configure DeepSpeed ZeRO optimizer
    "deepspeed_zero_optimization": {
        "stage" : 0|1|2|3
    },
    
    ## Options to configure PyTorch(FSDP) ZeRO optimizer
    "fsdp_zero_optimization": {
        "stage" : 0|1|2|3,
        "flatten_parameters" : True|False
    },
    
    ## merged_experts -- Enhance performance by using batched GEMM
    "merged_experts": True|False,

    ## Option to select NVIDIA Apex Optimization level
    "nvidia_apex_opt_level" : "O0"|"01"|"O2"|"O3",

    ## Option to enable FP16 processing inside an MoE Expert
    "fp16_mode" : True|False,

    ## Options to enable support for imbalanced inputs. This feature
    ## supports distributed training scenario where each rank could
    ## have distinct input size (for example sequence length) that
    ## could interfere with communication collectives that expect same
    ## input tensor size on each rank. The support is enabled by default.
    "imbalanced_input_support" : {
        "enabled" : True|False,

        ## Additional all-reduce is used to handle imbalanced inputs.
        ## Using MPI for this all-reduce elminates extra DtoH copy so
        ## this is enabled by default.
        "use_mpi" : True|False
    },

    ## Option to control checkpointing of experts using torch.checkpointing
    ## API. Default is False.
    "checkpoint_experts" : False|True

    ## Option to enable dynamic capacity feature to dynamically adjust capacity
    ## factor during Expert Parallel mode. By default this feature is disabled.
    "enable_dynamic_capacity" : False|True

    ## Option to enable Basa Layer Shuffling. Paper link: https://arxiv.org/abs/2103.16716
    "enable_base_layer_shuffling" : False|True

    ## Option to enable Tutel cumsum optimization
    "enable_tutel_cumsum" : False|True

    ## Option to enable expert weight calculation optimization
    "enable_expert_weight_calculation_optimization" : False|True
  }
"""
class moe_config:
    def __init__(self, options):
        self._options = {}
        if options is not None:
            self._options = options

    def enable_verbose(self):
        r"""enable_verbose
            Returns true if verbose mode is on
        """
        return self._options.get("verbose", False)

    def enable_deepspeed_zero_optimization(self):
        r"""enable_deepspeed_zero_optimization:
        Returns true if DeepSpeed ZeRO stage 1, 2 or 3 is selected.
        """
        ds = self._options.get("deepspeed_zero_optimization", {})
        stage = ds.get("stage", 0)
        if stage > 0:
            return True
        return False

    def enable_fsdp_zero_optimization(self):
        r"""enable_fsdp_zero_optimization:
        Returns true if FSDP ZeRO stage 1, 2 or 3 is selected.
        """
        ds = self._options.get("fsdp_zero_optimization", {})
        stage = ds.get("stage", 0)
        if stage > 0:
            return True
        return False

    def fsdp_flatten_parameters(self):
        r"""fsdp_flatten_parameters:
        Returns true if flatten parameters optimization is enabled in the FSDP.
        """
        ds = self._options.get("fsdp_zero_optimization", {})
        stage = ds.get("flatten_parameters", True)
        if stage > 0:
            return True
        return False

    def enable_merged_experts(self):
        r"""enable_merged_experts:
        Returns true if merged_experts optimization is enable. This optimization 
        used batched gemm to improve performance. 
        """
        return self._options.get("merged_experts", True)

    def nvidia_apex_opt_level(self):
        r"""nvidia_apex_opt_level:
        Return selected Nvidia Apex opt level
        """
        return self._options.get("nvidia_apex_opt_level", None)

    def fp16_mode(self):
        r"""fp16_mode
        Return true if fp16_mode is enabled. In this mode Expert computations are
        done in fp16
        """
        return self._options.get("fp16_mode", False)

    def support_imbalanced_input(self):
        r"""support_imbalanced_input
        Return true if support for imbalanced input is enabled
        """
        ds = self._options.get("imbalanced_input_support", {})
        return ds.get("enabled", True)

    def use_mpi_for_imbalanced_input(self):
        r"""use_mpi_for_imbalanced_input
        Return true if use of MPI is enabled to support imbalanced inputs.
        """
        ds = self._options.get("imbalanced_input_support", {})
        if ds.get("enabled", True) is True:
            return ds.get("use_mpi", True)
        return False

    def checkpoint_experts(self):
        r"""checkpoint_experts
        Return true if experts should be checkpointed using torch API.
        """
        return self._options.get("checkpoint_experts", False)

    def enable_dynamic_capacity(self):
        r"""enable_dynamic_capacity
        Return true if capacity factor should be dynamically adjusted
        """
        return self._options.get("enable_dynamic_capacity", False)

    def enable_base_layer_shuffling(self):
        r"""enable_base_layer_shuffling
        Return true if Base Layer Shuffling is enabled.
        """
        return self._options.get("enable_base_layer_shuffling", False)

    def enable_tutel_cumsum(self):
        r"""enable_tutel_cumsum
        Return true if Tutel cumsum kernel is enabled
        """
        return self._options.get("enable_tutel_cumsum", False)

    def enable_expert_weight_calculation_optimization(self):
        r"""enable_expert_weight_calculation_optimization
        Returntrue if expert weight calculation optimization is enabled
        """
        return self._options.get("enable_expert_weight_calculation_optimization", False)
