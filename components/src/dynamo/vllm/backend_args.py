# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo vLLM wrapper configuration ArgGroup."""

from typing import Optional

from dynamo.common.configuration.arg_group import ArgGroup
from dynamo.common.configuration.config_base import ConfigBase
from dynamo.common.configuration.utils import add_argument, add_negatable_bool_argument

from . import __version__


class DynamoVllmArgGroup(ArgGroup):
    """vLLM-specific Dynamo wrapper configuration (not native vLLM engine args)."""

    name = "dynamo-vllm"

    def add_arguments(self, parser) -> None:
        """Add Dynamo vLLM arguments to parser."""

        parser.add_argument(
            "--version", action="version", version=f"Dynamo Backend VLLM {__version__}"
        )
        g = parser.add_argument_group("Dynamo vLLM Options")

        add_negatable_bool_argument(
            g,
            flag_name="--is-prefill-worker",
            env_var="DYN_VLLM_IS_PREFILL_WORKER",
            default=False,
            help="Enable prefill functionality for this worker. Uses the provided namespace to construct dyn://namespace.prefill.generate",
        )

        add_negatable_bool_argument(
            g,
            flag_name="--is-decode-worker",
            env_var="DYN_VLLM_IS_DECODE_WORKER",
            default=False,
            help="Mark this as a decode worker which does not publish KV events",
        )

        add_negatable_bool_argument(
            g,
            flag_name="--use-vllm-tokenizer",
            env_var="DYN_VLLM_USE_TOKENIZER",
            default=False,
            help="Use vLLM's tokenizer for pre and post processing. This bypasses Dynamo's preprocessor and only v1/chat/completions will be available through the Dynamo frontend.",
        )

        add_argument(
            g,
            flag_name="--sleep-mode-level",
            env_var="DYN_VLLM_SLEEP_MODE_LEVEL",
            default=1,
            help="Sleep mode level (1=offload to CPU, 2=discard weights, 3=discard all).",
            choices=[1, 2, 3],
            arg_type=int,
        )

        # Multimodal
        add_negatable_bool_argument(
            g,
            flag_name="--multimodal-processor",
            env_var="DYN_VLLM_MULTIMODAL_PROCESSOR",
            default=False,
            help="Run as multimodal processor component for handling multimodal requests.",
        )
        add_negatable_bool_argument(
            g,
            flag_name="--ec-processor",
            env_var="DYN_VLLM_EC_PROCESSOR",
            default=False,
            help="Run as ECConnector processor (routes multimodal requests to encoder then PD workers).",
        )
        add_negatable_bool_argument(
            g,
            flag_name="--multimodal-encode-worker",
            env_var="DYN_VLLM_MULTIMODAL_ENCODE_WORKER",
            default=False,
            help="Run as multimodal encode worker component for processing images/videos.",
        )
        add_negatable_bool_argument(
            g,
            flag_name="--multimodal-worker",
            env_var="DYN_VLLM_MULTIMODAL_WORKER",
            default=False,
            help="Run as multimodal worker component for LLM inference with multimodal data.",
        )
        add_negatable_bool_argument(
            g,
            flag_name="--multimodal-decode-worker",
            env_var="DYN_VLLM_MULTIMODAL_DECODE_WORKER",
            default=False,
            help="Run as multimodal decode worker in disaggregated mode.",
        )
        add_negatable_bool_argument(
            g,
            flag_name="--multimodal-encode-prefill-worker",
            env_var="DYN_VLLM_MULTIMODAL_ENCODE_PREFILL_WORKER",
            default=False,
            help="Run as unified encode+prefill+decode worker for models requiring integrated image encoding (e.g., Llama 4).",
        )
        add_negatable_bool_argument(
            g,
            flag_name="--enable-multimodal",
            env_var="DYN_VLLM_ENABLE_MULTIMODAL",
            default=False,
            help="Enable multimodal processing. If not set, none of the multimodal components can be used.",
        )
        add_argument(
            g,
            flag_name="--mm-prompt-template",
            env_var="DYN_VLLM_MM_PROMPT_TEMPLATE",
            default="USER: <image>\n<prompt> ASSISTANT:",
            help=(
                "Different multi-modal models expect the prompt to contain different special media prompts. "
                "The processor will use this argument to construct the final prompt. "
                "User prompt will replace '<prompt>' in the provided template. "
                "For example, if the user prompt is 'please describe the image' and the prompt template is "
                "'USER: <image> <prompt> ASSISTANT:', the resulting prompt is "
                "'USER: <image> please describe the image ASSISTANT:'."
            ),
        )

        add_negatable_bool_argument(
            g,
            flag_name="--frontend-decoding",
            env_var="DYN_VLLM_FRONTEND_DECODING",
            default=False,
            help=(
                "Enable frontend decoding of multimodal images. "
                "When enabled, images are decoded in the Rust frontend and transferred to the backend via NIXL RDMA. "
                "Without this flag, images are decoded in the Python backend (default behavior)."
            ),
        )

        # vLLM-native encoder (ECConnector)
        add_negatable_bool_argument(
            g,
            flag_name="--vllm-native-encoder-worker",
            env_var="DYN_VLLM_NATIVE_ENCODER_WORKER",
            default=False,
            help="Run as vLLM-native encoder worker using ECConnector for encoder disaggregation (requires shared storage). The following flags only work when this flag is enabled: --ec-connector-backend, --ec-storage-path, --ec-extra-config, --ec-consumer-mode.",
        )
        add_argument(
            g,
            flag_name="--ec-connector-backend",
            env_var="DYN_VLLM_EC_CONNECTOR_BACKEND",
            default="ECExampleConnector",
            help="ECConnector implementation class for encoder disaggregation.",
        )
        add_argument(
            g,
            flag_name="--ec-storage-path",
            env_var="DYN_VLLM_EC_STORAGE_PATH",
            default=None,
            help="Storage path for ECConnector (required for ECExampleConnector, optional for other backends).",
        )
        add_argument(
            g,
            flag_name="--ec-extra-config",
            env_var="DYN_VLLM_EC_EXTRA_CONFIG",
            default=None,
            help="Additional ECConnector configuration as JSON string.",
        )
        add_negatable_bool_argument(
            g,
            flag_name="--ec-consumer-mode",
            env_var="DYN_VLLM_EC_CONSUMER_MODE",
            default=False,
            help="Configure as ECConnector consumer for receiving encoder embeddings (for PD workers).",
        )

        # vLLM-Omni
        add_negatable_bool_argument(
            g,
            flag_name="--omni",
            env_var="DYN_VLLM_OMNI",
            default=False,
            help="Run as vLLM-Omni worker for multi-stage pipelines (supports text-to-text, text-to-image, etc.).",
        )
        add_argument(
            g,
            flag_name="--stage-configs-path",
            env_var="DYN_VLLM_STAGE_CONFIGS_PATH",
            default=None,
            help="Path to vLLM-Omni stage configuration YAML file for --omni mode (optional).",
        )

        # ModelExpress P2P
        add_argument(
            g,
            flag_name="--model-express-url",
            env_var="MODEL_EXPRESS_URL",
            default=None,
            help="ModelExpress P2P server URL (e.g., http://mx-server:8080). "
            "Required when using --load-format=mx-source or --load-format=mx-target.",
        )


# @dataclass()
class DynamoVllmConfig(ConfigBase):
    """Configuration for Dynamo vLLM wrapper (vLLM-specific only). All fields optional."""

    is_prefill_worker: bool
    is_decode_worker: bool
    use_vllm_tokenizer: bool
    sleep_mode_level: int

    # Multimodal
    multimodal_processor: bool
    ec_processor: bool
    multimodal_encode_worker: bool
    multimodal_worker: bool
    multimodal_decode_worker: bool
    multimodal_encode_prefill_worker: bool
    enable_multimodal: bool
    mm_prompt_template: str
    frontend_decoding: bool

    # vLLM-native encoder (ECConnector)
    vllm_native_encoder_worker: bool
    ec_connector_backend: str
    ec_storage_path: Optional[str] = None
    ec_extra_config: Optional[str] = None
    ec_consumer_mode: bool

    # vLLM-Omni
    omni: bool
    stage_configs_path: Optional[str] = None

    # ModelExpress P2P
    model_express_url: Optional[str] = None

    def validate(self) -> None:
        """Validate vLLM wrapper configuration."""
        self._validate_prefill_decode_exclusive()
        self._validate_multimodal_role_exclusivity()
        self._validate_multimodal_requires_flag()
        self._validate_ec_connector_storage()
        self._validate_omni_stage_config()

    def _validate_prefill_decode_exclusive(self) -> None:
        """Ensure at most one of is_prefill_worker and is_decode_worker is set."""
        if self.is_prefill_worker and self.is_decode_worker:
            raise ValueError(
                "Cannot set both --is-prefill-worker and --is-decode-worker"
            )

    def _count_multimodal_roles(self) -> int:
        """Return the number of multimodal roles set (0 or 1 allowed)."""
        return sum(
            [
                bool(self.multimodal_processor),
                bool(self.ec_processor),
                bool(self.multimodal_encode_worker),
                bool(self.multimodal_worker),
                bool(self.multimodal_decode_worker),
                bool(self.multimodal_encode_prefill_worker),
                bool(self.vllm_native_encoder_worker),
            ]
        )

    def _validate_multimodal_role_exclusivity(self) -> None:
        """Ensure only one multimodal role is set at a time."""
        if self._count_multimodal_roles() > 1:
            raise ValueError(
                "Only one multimodal role can be set at a time: "
                "multimodal-processor, ec-processor, multimodal-encode-worker, "
                "multimodal-worker, multimodal-decode-worker, "
                "multimodal-encode-prefill-worker, vllm-native-encoder-worker"
            )

    def _validate_multimodal_requires_flag(self) -> None:
        """Require --enable-multimodal when any multimodal role is set."""
        if self._count_multimodal_roles() == 1 and not self.enable_multimodal:
            raise ValueError(
                "Use --enable-multimodal when enabling any multimodal component"
            )

    def _validate_ec_connector_storage(self) -> None:
        """Require ec_storage_path when using ECExampleConnector backend."""
        if self.vllm_native_encoder_worker:
            if (
                self.ec_connector_backend == "ECExampleConnector"
                and not self.ec_storage_path
            ):
                raise ValueError(
                    "--ec-storage-path is required when using ECExampleConnector backend. "
                    "Specify a shared storage path for encoder cache."
                )

    def _validate_omni_stage_config(self) -> None:
        """Require stage_configs_path when using --omni."""
        if self.stage_configs_path and not self.omni:
            raise ValueError(
                "--stage-configs-path is only allowed when using --omni. "
                "Specify a YAML file containing stage configurations for the multi-stage pipeline."
            )
