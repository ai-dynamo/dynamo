# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo TRT-LLM backend configuration ArgGroup."""

from typing import Optional

from tensorrt_llm.llmapi import BuildConfig

from dynamo.common.configuration.arg_group import ArgGroup
from dynamo.common.configuration.config_base import ConfigBase
from dynamo.common.configuration.utils import add_argument, add_negatable_bool_argument

from . import __version__
from .constants import DisaggregationMode, Modality

DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


class DynamoTrtllmArgGroup(ArgGroup):
    """TensorRT-LLM-specific Dynamo wrapper configuration."""

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--version",
            action="version",
            version=f"Dynamo Backend TRTLLM {__version__}",
        )
        g = parser.add_argument_group("Dynamo TRT-LLM Options")

        add_argument(
            g,
            flag_name="--model",
            env_var="DYN_TRTLLM_MODEL",
            default=DEFAULT_MODEL,
            obsolete_flag="--model-path",
            help=("Path to disk model or HuggingFace model identifier to load. "),
        )
        add_argument(
            g,
            flag_name="--served-model-name",
            env_var="DYN_TRTLLM_SERVED_MODEL_NAME",
            default=None,
            help="Name to serve the model under. Defaults to deriving it from model path.",
        )
        add_argument(
            g,
            flag_name="--tensor-parallel-size",
            env_var="DYN_TRTLLM_TENSOR_PARALLEL_SIZE",
            default=1,
            arg_type=int,
            help="Tensor parallelism size.",
        )
        add_argument(
            g,
            flag_name="--pipeline-parallel-size",
            env_var="DYN_TRTLLM_PIPELINE_PARALLEL_SIZE",
            default=1,
            arg_type=int,
            help="Pipeline parallelism size.",
        )
        add_argument(
            g,
            flag_name="--expert-parallel-size",
            env_var="DYN_TRTLLM_EXPERT_PARALLEL_SIZE",
            default=None,
            arg_type=int,
            help="Expert parallelism size.",
        )
        add_negatable_bool_argument(
            g,
            flag_name="--enable-attention-dp",
            env_var="DYN_TRTLLM_ENABLE_ATTENTION_DP",
            default=False,
            help="Enable attention data parallelism. When enabled, attention_dp_size equals tensor_parallel_size.",
        )
        add_argument(
            g,
            flag_name="--kv-block-size",
            env_var="DYN_TRTLLM_KV_BLOCK_SIZE",
            default=32,
            arg_type=int,
            help="Size of a KV cache block.",
        )
        add_argument(
            g,
            flag_name="--gpus-per-node",
            env_var="DYN_TRTLLM_GPUS_PER_NODE",
            default=None,
            arg_type=int,
            help="Number of GPUs per node. If not provided, inferred from the environment.",
        )
        add_argument(
            g,
            flag_name="--max-batch-size",
            env_var="DYN_TRTLLM_MAX_BATCH_SIZE",
            default=BuildConfig.model_fields["max_batch_size"].default,
            arg_type=int,
            help="Maximum number of requests that the engine can schedule.",
        )
        add_argument(
            g,
            flag_name="--max-num-tokens",
            env_var="DYN_TRTLLM_MAX_NUM_TOKENS",
            default=BuildConfig.model_fields["max_num_tokens"].default,
            arg_type=int,
            help="Maximum number of batched input tokens after padding is removed in each batch.",
        )
        add_argument(
            g,
            flag_name="--max-seq-len",
            env_var="DYN_TRTLLM_MAX_SEQ_LEN",
            default=BuildConfig.model_fields["max_seq_len"].default,
            arg_type=int,
            help="Maximum total length of one request, including prompt and outputs. If unspecified, the value is deduced from the model config.",
        )
        add_argument(
            g,
            flag_name="--max-beam-width",
            env_var="DYN_TRTLLM_MAX_BEAM_WIDTH",
            default=BuildConfig.model_fields["max_beam_width"].default,
            arg_type=int,
            help="Maximum number of beams for beam search decoding.",
        )
        add_argument(
            g,
            flag_name="--free-gpu-memory-fraction",
            env_var="DYN_TRTLLM_FREE_GPU_MEMORY_FRACTION",
            default=0.9,
            arg_type=float,
            help="Free GPU memory fraction reserved for KV Cache,  after model weights and buffers are allocated.",
        )
        add_argument(
            g,
            flag_name="--extra-engine-args",
            env_var="DYN_TRTLLM_EXTRA_ENGINE_ARGS",
            default="",
            help="Path to a YAML file containing additional keyword arguments to pass to the TRTLLM engine.",
        )
        add_argument(
            g,
            flag_name="--override-engine-args",
            env_var="DYN_TRTLLM_OVERRIDE_ENGINE_ARGS",
            default="",
            help="Python dictionary string to override specific engine arguments from the YAML file. "
            'Example: \'{"tensor_parallel_size": 2, "kv_cache_config": {"enable_block_reuse": false}}\'',
        )
        add_negatable_bool_argument(
            g,
            flag_name="--publish-events-and-metrics",
            env_var="DYN_TRTLLM_PUBLISH_EVENTS_AND_METRICS",
            default=False,
            help="If set, publish events and metrics to Dynamo components.",
        )
        add_negatable_bool_argument(
            g,
            flag_name="--disable-request-abort",
            env_var="DYN_TRTLLM_DISABLE_REQUEST_ABORT",
            default=True,
            help="Disable calling abort() on the TRT-LLM engine when a request is cancelled.",
        )
        add_argument(
            g,
            flag_name="--disaggregation-mode",
            env_var="DYN_TRTLLM_DISAGGREGATION_MODE",
            default=DisaggregationMode.AGGREGATED.value,
            choices=[mode.value for mode in DisaggregationMode],
            help="Mode to use for disaggregation.",
        )
        add_argument(
            g,
            flag_name="--modality",
            env_var="DYN_TRTLLM_MODALITY",
            default=Modality.TEXT.value,
            choices=[m.value for m in Modality],
            help="Modality to use for the model.",
        )
        add_argument(
            g,
            flag_name="--encode-endpoint",
            env_var="DYN_TRTLLM_ENCODE_ENDPOINT",
            default="",
            help="Endpoint (in 'dyn://namespace.component.endpoint' format) for the encode worker.",
        )
        add_argument(
            g,
            flag_name="--allowed-local-media-path",
            env_var="DYN_TRTLLM_ALLOWED_LOCAL_MEDIA_PATH",
            default="",
            help="Path to a directory that is allowed to be accessed by the model.",
        )
        add_argument(
            g,
            flag_name="--max-file-size-mb",
            env_var="DYN_TRTLLM_MAX_FILE_SIZE_MB",
            default=50,
            arg_type=int,
            help="Maximum size of downloadable embedding files/Image URLs.",
        )

        diffusion_group = parser.add_argument_group(
            "Diffusion Options [Experimental]",
            "Options for video_diffusion modality",
        )
        add_argument(
            diffusion_group,
            flag_name="--default-height",
            env_var="DYN_TRTLLM_DEFAULT_HEIGHT",
            default=480,
            arg_type=int,
            help="Default video/image height in pixels.",
        )
        add_argument(
            diffusion_group,
            flag_name="--default-width",
            env_var="DYN_TRTLLM_DEFAULT_WIDTH",
            default=832,
            arg_type=int,
            help="Default video/image width in pixels.",
        )
        add_argument(
            diffusion_group,
            flag_name="--default-num-frames",
            env_var="DYN_TRTLLM_DEFAULT_NUM_FRAMES",
            default=81,
            arg_type=int,
            help="Default number of frames for video generation.",
        )
        add_argument(
            diffusion_group,
            flag_name="--default-num-inference-steps",
            env_var="DYN_TRTLLM_DEFAULT_NUM_INFERENCE_STEPS",
            default=50,
            arg_type=int,
            help="Default number of inference steps.",
        )
        add_argument(
            diffusion_group,
            flag_name="--default-guidance-scale",
            env_var="DYN_TRTLLM_DEFAULT_GUIDANCE_SCALE",
            default=5.0,
            arg_type=float,
            help="Default CFG guidance scale.",
        )
        add_negatable_bool_argument(
            diffusion_group,
            flag_name="--enable-teacache",
            env_var="DYN_TRTLLM_ENABLE_TEACACHE",
            default=False,
            help="Enable TeaCache optimization for faster generation.",
        )
        add_argument(
            diffusion_group,
            flag_name="--teacache-thresh",
            env_var="DYN_TRTLLM_TEACACHE_THRESH",
            default=0.2,
            arg_type=float,
            help="TeaCache threshold.",
        )
        add_argument(
            diffusion_group,
            flag_name="--attn-type",
            env_var="DYN_TRTLLM_ATTN_TYPE",
            default="default",
            choices=["default", "sage-attn", "sparse-videogen", "sparse-videogen2"],
            help="Attention type for diffusion models.",
        )
        add_argument(
            diffusion_group,
            flag_name="--linear-type",
            env_var="DYN_TRTLLM_LINEAR_TYPE",
            default="default",
            choices=[
                "default",
                "trtllm-fp8-blockwise",
                "trtllm-fp8-per-tensor",
                "trtllm-nvfp4",
            ],
            help="Linear type for quantization.",
        )
        add_negatable_bool_argument(
            diffusion_group,
            flag_name="--disable-torch-compile",
            env_var="DYN_TRTLLM_DISABLE_TORCH_COMPILE",
            default=False,
            help="Disable torch.compile optimization.",
        )
        add_argument(
            diffusion_group,
            flag_name="--torch-compile-mode",
            env_var="DYN_TRTLLM_TORCH_COMPILE_MODE",
            default="default",
            choices=["default", "reduce-overhead", "max-autotune"],
            help="torch.compile mode.",
        )
        add_argument(
            diffusion_group,
            flag_name="--dit-dp-size",
            env_var="DYN_TRTLLM_DIT_DP_SIZE",
            default=1,
            arg_type=int,
            help="Data parallel size for DiT.",
        )
        add_argument(
            diffusion_group,
            flag_name="--dit-tp-size",
            env_var="DYN_TRTLLM_DIT_TP_SIZE",
            default=1,
            arg_type=int,
            help="Tensor parallel size for DiT.",
        )
        add_argument(
            diffusion_group,
            flag_name="--dit-ulysses-size",
            env_var="DYN_TRTLLM_DIT_ULYSSES_SIZE",
            default=1,
            arg_type=int,
            help="Ulysses parallel size for DiT.",
        )
        add_argument(
            diffusion_group,
            flag_name="--dit-ring-size",
            env_var="DYN_TRTLLM_DIT_RING_SIZE",
            default=1,
            arg_type=int,
            help="Ring parallel size for DiT.",
        )
        add_argument(
            diffusion_group,
            flag_name="--dit-cfg-size",
            env_var="DYN_TRTLLM_DIT_CFG_SIZE",
            default=1,
            arg_type=int,
            help="CFG parallel size for DiT.",
        )
        add_argument(
            diffusion_group,
            flag_name="--dit-fsdp-size",
            env_var="DYN_TRTLLM_DIT_FSDP_SIZE",
            default=1,
            arg_type=int,
            help="FSDP size for DiT.",
        )
        add_negatable_bool_argument(
            diffusion_group,
            flag_name="--enable-async-cpu-offload",
            env_var="DYN_TRTLLM_ENABLE_ASYNC_CPU_OFFLOAD",
            default=False,
            help="Enable async CPU offload for memory efficiency.",
        )


class DynamoTrtllmConfig(ConfigBase):
    """Configuration for Dynamo TRT-LLM backend-specific options."""

    model: str
    served_model_name: Optional[str] = None

    tensor_parallel_size: int
    pipeline_parallel_size: int
    expert_parallel_size: Optional[int]
    enable_attention_dp: bool
    kv_block_size: int
    gpus_per_node: Optional[int] = None
    max_batch_size: int
    max_num_tokens: int
    max_seq_len: int
    max_beam_width: int
    free_gpu_memory_fraction: float
    extra_engine_args: str
    override_engine_args: str
    publish_events_and_metrics: bool
    disable_request_abort: bool

    disaggregation_mode: DisaggregationMode
    modality: Modality
    encode_endpoint: str
    allowed_local_media_path: str
    max_file_size_mb: int

    default_height: int
    default_width: int
    default_num_frames: int
    default_num_inference_steps: int
    default_guidance_scale: float
    enable_teacache: bool
    teacache_thresh: float
    attn_type: str
    linear_type: str
    disable_torch_compile: bool
    torch_compile_mode: str
    dit_dp_size: int
    dit_tp_size: int
    dit_ulysses_size: int
    dit_ring_size: int
    dit_cfg_size: int
    dit_fsdp_size: int
    enable_async_cpu_offload: bool

    def validate(self) -> None:
        if isinstance(self.disaggregation_mode, str):
            self.disaggregation_mode = DisaggregationMode(self.disaggregation_mode)
        if isinstance(self.modality, str):
            self.modality = Modality(self.modality)
        if not self.served_model_name:
            self.served_model_name = None
