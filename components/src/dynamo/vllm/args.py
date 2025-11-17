# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
import os
import socket
from typing import Any, Dict, Optional

from vllm.config import KVTransferConfig
from vllm.distributed.kv_events import KVEventsConfig
from vllm.engine.arg_utils import AsyncEngineArgs

try:
    from vllm.utils import FlexibleArgumentParser
except ImportError:
    from vllm.utils.argparse_utils import FlexibleArgumentParser

from dynamo._core import get_reasoning_parser_names, get_tool_parser_names
from dynamo.common.config_dump import add_config_dump_args, register_encoder

from . import __version__, envs

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
VALID_CONNECTORS = {"nixl", "lmcache", "kvbm", "null", "none"}

# Global LMCache configuration - initialize once on module import
ENABLE_LMCACHE = os.getenv("ENABLE_LMCACHE", "0").lower() in ("1", "true", "yes")


class Config:
    """Command line parameters or defaults"""

    # dynamo specific
    namespace: str
    component: str
    endpoint: str
    is_prefill_worker: bool
    is_decode_worker: bool
    migration_limit: int = 0
    custom_jinja_template: Optional[str] = None
    store_kv: str
    request_plane: str

    # mirror vLLM
    model: str
    served_model_name: Optional[str]

    # rest vLLM args
    engine_args: AsyncEngineArgs

    # Connector list from CLI
    connector_list: Optional[list] = None

    # tool and reasoning parser info
    tool_call_parser: Optional[str] = None
    reasoning_parser: Optional[str] = None

    # multimodal options
    multimodal_processor: bool = False
    multimodal_encode_worker: bool = False
    multimodal_worker: bool = False
    multimodal_decode_worker: bool = False
    multimodal_encode_prefill_worker: bool = False
    mm_prompt_template: str = "USER: <image>\n<prompt> ASSISTANT:"
    # dump config to file
    dump_config_to: Optional[str] = None

    def has_connector(self, connector_name: str) -> bool:
        """
        Check if a specific connector is enabled.

        Args:
            connector_name: Name of the connector to check (e.g., "kvbm", "nixl")

        Returns:
            True if the connector is in the connector list, False otherwise
        """
        return self.connector_list is not None and connector_name in self.connector_list


@register_encoder(Config)
def _preprocess_for_encode_config(config: Config) -> Dict[str, Any]:
    return config.__dict__


def parse_args() -> Config:
    parser = FlexibleArgumentParser(
        description="vLLM server integrated with Dynamo LLM."
    )
    parser.add_argument(
        "--version", action="version", version=f"Dynamo Backend VLLM {__version__}"
    )
    parser.add_argument(
        "--is-prefill-worker",
        action="store_true",
        help="Enable prefill functionality for this worker. Uses the provided namespace to construct dyn://namespace.prefill.generate",
    )
    parser.add_argument(
        "--is-decode-worker",
        action="store_true",
        help="Mark this as a decode worker which does not publish KV events.",
    )
    parser.add_argument(
        "--migration-limit",
        type=int,
        default=0,
        help="Maximum number of times a request may be migrated to a different engine worker. The number may be overridden by the engine.",
    )
    parser.add_argument(
        "--connector",
        nargs="*",
        default=["nixl"],
        help="List of connectors to use in order (e.g., --connector nixl lmcache). "
        "Options: nixl, lmcache, kvbm, null, none. Default: nixl. Order will be preserved in MultiConnector.",
    )
    # To avoid name conflicts with different backends, adopted prefix "dyn-" for dynamo specific args
    parser.add_argument(
        "--dyn-tool-call-parser",
        type=str,
        default=None,
        choices=get_tool_parser_names(),
        help="Tool call parser name for the model.",
    )
    parser.add_argument(
        "--dyn-reasoning-parser",
        type=str,
        default=None,
        choices=get_reasoning_parser_names(),
        help="Reasoning parser name for the model. If not specified, no reasoning parsing is performed.",
    )
    parser.add_argument(
        "--custom-jinja-template",
        type=str,
        default=None,
        help="Path to a custom Jinja template file to override the model's default chat template. This template will take precedence over any template found in the model repository.",
    )
    parser.add_argument(
        "--multimodal-processor",
        action="store_true",
        help="Run as multimodal processor component for handling multimodal requests",
    )
    parser.add_argument(
        "--multimodal-encode-worker",
        action="store_true",
        help="Run as multimodal encode worker component for processing images/videos",
    )
    parser.add_argument(
        "--multimodal-worker",
        action="store_true",
        help="Run as multimodal worker component for LLM inference with multimodal data",
    )
    parser.add_argument(
        "--multimodal-decode-worker",
        action="store_true",
        help="Run as multimodal decode worker in disaggregated mode",
    )
    parser.add_argument(
        "--multimodal-encode-prefill-worker",
        action="store_true",
        help="Run as unified encode+prefill+decode worker for models requiring integrated image encoding (e.g., Llama 4)",
    )
    parser.add_argument(
        "--mm-prompt-template",
        type=str,
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
    parser.add_argument(
        "--store-kv",
        type=str,
        choices=["etcd", "file", "mem"],
        default=os.environ.get("DYN_STORE_KV", "etcd"),
        help="Which key-value backend to use: etcd, mem, file. Etcd uses the ETCD_* env vars (e.g. ETCD_ENPOINTS) for connection details. File uses root dir from env var DYN_FILE_KV or defaults to $TMPDIR/dynamo_store_kv.",
    )
    parser.add_argument(
        "--request-plane",
        type=str,
        choices=["nats", "http", "tcp"],
        default=os.environ.get("DYN_REQUEST_PLANE", "nats"),
        help="Determines how requests are distributed from routers to workers. 'tcp' is fastest [nats|http|tcp]",
    )
    add_config_dump_args(parser)

    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    engine_args = AsyncEngineArgs.from_cli_args(args)

    if engine_args.enable_prefix_caching is None:
        logger.debug(
            "--enable-prefix-caching or --no-enable-prefix-caching not specified. Defaulting to True (vLLM v1 default behavior)"
        )
        engine_args.enable_prefix_caching = True

    config = Config()
    config.model = args.model
    if args.served_model_name:
        assert (
            len(args.served_model_name) <= 1
        ), "We do not support multiple model names."
        config.served_model_name = args.served_model_name[0]
    else:
        # This becomes an `Option` on the Rust side
        config.served_model_name = None

    config.namespace = os.environ.get("DYN_NAMESPACE", "dynamo")

    # Check multimodal role exclusivity
    mm_flags = (
        int(bool(args.multimodal_processor))
        + int(bool(args.multimodal_encode_worker))
        + int(bool(args.multimodal_worker))
        + int(bool(args.multimodal_decode_worker))
        + int(bool(args.multimodal_encode_prefill_worker))
    )
    if mm_flags > 1:
        raise ValueError(
            "Use only one of --multimodal-processor, --multimodal-encode-worker, --multimodal-worker, --multimodal-decode-worker, or --multimodal-encode-prefill-worker"
        )

    # Set component and endpoint based on worker type
    if args.multimodal_processor:
        config.component = "processor"
        config.endpoint = "generate"
    elif args.multimodal_encode_worker:
        config.component = "encoder"
        config.endpoint = "generate"
    elif args.multimodal_encode_prefill_worker:
        config.component = "encoder"
        config.endpoint = "generate"
    elif args.multimodal_decode_worker:
        # Uses "decoder" component name because prefill worker connects to "decoder"
        # (prefill uses "backend" to receive from encoder)
        config.component = "decoder"
        config.endpoint = "generate"
    elif args.multimodal_worker and args.is_prefill_worker:
        # Multimodal prefill worker stays as "backend" to maintain encoder connection
        config.component = "backend"
        config.endpoint = "generate"
    elif args.is_prefill_worker:
        config.component = "prefill"
        config.endpoint = "generate"
    else:
        config.component = "backend"
        config.endpoint = "generate"

    config.engine_args = engine_args
    config.is_prefill_worker = args.is_prefill_worker
    config.is_decode_worker = args.is_decode_worker
    config.migration_limit = args.migration_limit
    config.tool_call_parser = args.dyn_tool_call_parser
    config.reasoning_parser = args.dyn_reasoning_parser
    config.custom_jinja_template = args.custom_jinja_template
    config.multimodal_processor = args.multimodal_processor
    config.multimodal_encode_worker = args.multimodal_encode_worker
    config.multimodal_worker = args.multimodal_worker
    config.multimodal_decode_worker = args.multimodal_decode_worker
    config.multimodal_encode_prefill_worker = args.multimodal_encode_prefill_worker
    config.mm_prompt_template = args.mm_prompt_template
    config.store_kv = args.store_kv
    config.request_plane = args.request_plane

    # Validate custom Jinja template file exists if provided
    if config.custom_jinja_template is not None:
        # Expand environment variables and user home (~) before validation
        expanded_template_path = os.path.expanduser(
            os.path.expandvars(config.custom_jinja_template)
        )
        config.custom_jinja_template = expanded_template_path
        if not os.path.isfile(expanded_template_path):
            raise FileNotFoundError(
                f"Custom Jinja template file not found: {expanded_template_path}. "
                f"Please ensure the file exists and the path is correct."
            )

    # Check for conflicting flags
    has_kv_transfer_config = (
        hasattr(engine_args, "kv_transfer_config")
        and engine_args.kv_transfer_config is not None
    )
    has_connector_flag = args.connector is not None

    if has_kv_transfer_config and has_connector_flag:
        raise ValueError(
            "Cannot specify both --kv-transfer-config and --connector flags"
        )

    if has_connector_flag:
        normalized = [c.lower() for c in args.connector]

        invalid = [c for c in normalized if c not in VALID_CONNECTORS]
        if invalid:
            raise ValueError(
                f"Invalid connector(s): {', '.join(invalid)}. Valid options are: {', '.join(sorted(VALID_CONNECTORS))}"
            )

        if "none" in normalized or "null" in normalized:
            if len(normalized) > 1:
                raise ValueError(
                    "'none' and 'null' cannot be combined with other connectors"
                )
            config.connector_list = []
        else:
            config.connector_list = normalized

    if config.engine_args.block_size is None:
        config.engine_args.block_size = 16
        logger.debug(
            f"Setting reasonable default of {config.engine_args.block_size} for block_size"
        )

    config.dump_config_to = args.dump_config_to

    return config


def create_kv_events_config(config: Config) -> Optional[KVEventsConfig]:
    """Create KVEventsConfig for prefix caching if needed."""
    if config.is_decode_worker:
        logger.info(
            f"Decode worker detected (is_decode_worker={config.is_decode_worker}): "
            f"kv_events_config disabled (decode workers don't publish KV events)"
        )
        return None

    # If prefix caching is not enabled, no events config needed
    if not config.engine_args.enable_prefix_caching:
        logger.info("No kv_events_config required: prefix caching is disabled")
        return None

    # There is a bug with KV events publishing when LORA is enabled.
    # This is fixed in https://github.com/vllm-project/vllm/pull/27728 but not released yet.
    # remove below check once new vLLM version is released with the fix.
    if config.engine_args.enable_lora:
        if config.engine_args.kv_events_config is None:
            # No explicit kv events config provided by user, we'll disable kv cache because LoRA is enabled and its not supported yet.
            return None
        else:
            # User provided their own kv events config and it'll not work when LoRA is enabled.
            message = (
                "KV events doesn't work when LoRA is enabled due to upstream vLLM bug. "
                "Please see https://github.com/vllm-project/vllm/pull/27728."
                "For now, either disable lora or dont use explicit kv envents config."
                "Dont set both --kv-events-config and --enable-lora in vllm command line args."
            )
            logger.error(message)
            raise ValueError(message)

    # If user provided their own config, use that
    if c := getattr(config.engine_args, "kv_events_config"):
        logger.info(f"Using user-provided kv_events_config {c}")
        return c

    # Create default events config for prefix caching
    port = envs.DYN_VLLM_KV_EVENT_PORT
    logger.info(
        f"Using env-var DYN_VLLM_KV_EVENT_PORT={port} to create kv_events_config"
    )
    dp_rank = config.engine_args.data_parallel_rank or 0

    return KVEventsConfig(
        enable_kv_cache_events=True,
        publisher="zmq",
        endpoint=f"tcp://*:{port - dp_rank}",  # vLLM will iterate dp_rank for us, so we need to subtract it out TODO: fix in vLLM
    )


def create_kv_transfer_config(config: Config) -> Optional[KVTransferConfig]:
    """Create KVTransferConfig based on user config or connector list.

    Handles logging and returns the appropriate config or None.
    """
    has_user_kv_config = (
        hasattr(config.engine_args, "kv_transfer_config")
        and config.engine_args.kv_transfer_config is not None
    )

    if has_user_kv_config:
        logger.info("Using user-provided kv_transfer_config from --kv-transfer-config")
        return None  # Let vLLM use the user's config

    # No connector list or empty list means no config
    if not config.connector_list:
        logger.info("Using vLLM defaults for kv_transfer_config")
        return None

    logger.info(f"Creating kv_transfer_config from --connector {config.connector_list}")

    # Create connector configs in specified order
    multi_connectors = []
    for connector in config.connector_list:
        if connector == "lmcache":
            connector_cfg = {"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"}
        elif connector == "nixl":
            connector_cfg = {"kv_connector": "NixlConnector", "kv_role": "kv_both"}
        elif connector == "kvbm":
            connector_cfg = {
                "kv_connector": "DynamoConnector",
                "kv_connector_module_path": "kvbm.vllm_integration.connector",
                "kv_role": "kv_both",
            }
        multi_connectors.append(connector_cfg)

    # For single connector, return direct config
    if len(multi_connectors) == 1:
        cfg = multi_connectors[0]
        return KVTransferConfig(**cfg)

    # For multiple connectors, use PdConnector
    return KVTransferConfig(
        kv_connector="PdConnector",
        kv_role="kv_both",
        kv_connector_extra_config={"connectors": multi_connectors},
        kv_connector_module_path="kvbm.vllm_integration.connector",
    )


def overwrite_args(config):
    """Set vLLM defaults for Dynamo."""

    if config.has_connector("nixl"):
        ensure_side_channel_host()

    defaults = {
        "task": "generate",
        # As of vLLM >=0.10.0 the engine unconditionally calls
        # `sampling_params.update_from_tokenizer(...)`, so we can no longer
        # skip tokenizer initialisation.  Setting this to **False** avoids
        # a NoneType error when the processor accesses the tokenizer.
        "skip_tokenizer_init": False,
        "disable_log_requests": True,
        "disable_log_stats": False,
    }

    kv_transfer_config = create_kv_transfer_config(config)
    if kv_transfer_config:
        defaults["kv_transfer_config"] = kv_transfer_config

    defaults["kv_events_config"] = create_kv_events_config(config)
    logger.info(
        f"Using kv_events_config for publishing vLLM kv events over zmq: {defaults['kv_events_config']}"
    )

    logger.debug("Setting Dynamo defaults for vLLM")
    for key, value in defaults.items():
        if hasattr(config.engine_args, key):
            setattr(config.engine_args, key, value)
            logger.debug(f" engine_args.{key} = {value}")
        else:
            raise ValueError(f"{key} not found in AsyncEngineArgs from vLLM.")


def get_host_ip() -> str:
    """Get the IP address of the host for side-channel coordination."""
    try:
        host_name = socket.gethostname()
    except socket.error as exc:
        logger.warning("Failed to get hostname: %s, falling back to 127.0.0.1", exc)
        return "127.0.0.1"

    try:
        host_ip = socket.gethostbyname(host_name)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as test_socket:
            test_socket.bind((host_ip, 0))
        return host_ip
    except socket.gaierror as exc:
        logger.warning(
            "Hostname %s cannot be resolved: %s, falling back to 127.0.0.1",
            host_name,
            exc,
        )
        return "127.0.0.1"
    except socket.error as exc:
        logger.warning(
            "Hostname %s is not usable for binding: %s, falling back to 127.0.0.1",
            host_name,
            exc,
        )
        return "127.0.0.1"


def ensure_side_channel_host():
    """Ensure the NIXL side-channel host is available without overriding user settings."""

    existing_host = os.getenv("VLLM_NIXL_SIDE_CHANNEL_HOST")
    if existing_host:
        logger.debug(
            "Preserving existing VLLM_NIXL_SIDE_CHANNEL_HOST=%s", existing_host
        )
        return

    host_ip = get_host_ip()
    os.environ["VLLM_NIXL_SIDE_CHANNEL_HOST"] = host_ip
    logger.debug("Set VLLM_NIXL_SIDE_CHANNEL_HOST to %s", host_ip)
