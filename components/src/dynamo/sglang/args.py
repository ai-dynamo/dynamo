# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import contextlib
import json
import logging
import os
import socket
import sys
import tempfile
import warnings
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import yaml
from sglang.srt.server_args import ServerArgs
from sglang.srt.server_args_config_parser import ConfigArgumentMerger

from dynamo.common.config_dump import register_encoder
from dynamo.common.configuration.groups import DynamoRuntimeConfig
from dynamo.common.configuration.groups.router_args import (
    WorkerRouterConfig,
    parse_worker_router_config,
    register_worker_router_help,
)
from dynamo.common.configuration.groups.runtime_args import DynamoRuntimeArgGroup
from dynamo.common.configuration.utils import split_served_model_names
from dynamo.common.constants import DisaggregationMode
from dynamo.common.model_fetch import fetch_model
from dynamo.common.snapshot.lifecycle import (
    configure_snapshot_capture_env,
    is_snapshot_enabled,
)
from dynamo.common.utils.runtime import parse_endpoint
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.sglang._compat import ensure_sglang_tensor_image_size
from dynamo.sglang.backend_args import DynamoSGLangArgGroup, DynamoSGLangConfig
from dynamo.sglang.diffusion_args import build_diffusion_parser, parse_diffusion_args

configure_dynamo_logging()
PREFILL_DECODE_DISAGGREGATION_MODE = "pd"


class DynamoConfig(DynamoRuntimeConfig, DynamoSGLangConfig):
    """Combined configuration container for SGLang server and Dynamo args."""

    component: str
    diffusion_worker: bool = False
    # Whether this worker publishes KV events. Distinct from the router-side
    # `use_kv_events` on `router_advertisement`, which means the router
    # subscribes to them -- the reason the two live on separate objects.
    use_kv_events: bool = False
    # Routing this worker set advertises in its model card; None inherits the
    # frontend's configuration.
    router_advertisement: Optional[WorkerRouterConfig] = None

    def validate(self) -> None:
        DynamoRuntimeConfig.validate(self)
        DynamoSGLangConfig.validate(self)


class Config:
    """Combined configuration container for SGLang server and Dynamo args."""

    def __init__(self, server_args: ServerArgs, dynamo_args: DynamoConfig) -> None:
        self.server_args = server_args
        self.dynamo_args = dynamo_args
        self.serving_mode = self._set_serving_strategy()

    def _set_serving_strategy(self):
        if self.server_args.disaggregation_mode == "null":
            return DisaggregationMode.AGGREGATED
        elif self.server_args.disaggregation_mode == "prefill":
            return DisaggregationMode.PREFILL
        elif self.server_args.disaggregation_mode == "decode":
            return DisaggregationMode.DECODE
        else:
            return DisaggregationMode.AGGREGATED


def _unsupported_fpm_trace_role(dynamo_config: DynamoConfig) -> Optional[str]:
    """Return the worker role when the selected path does not create an FPM relay."""
    if is_snapshot_enabled():
        return "snapshot"
    if dynamo_config.embedding_worker:
        return "embedding"
    if (
        dynamo_config.multimodal_encode_worker
        or dynamo_config.multimodal_worker
        or dynamo_config.dedicated_mm_encoder
    ):
        return "dedicated multimodal"
    if dynamo_config.image_diffusion_worker:
        return "image diffusion"
    if dynamo_config.video_generation_worker:
        return "video generation"
    return None


def _forward_pass_metrics_source(dynamo_config: DynamoConfig) -> Optional[str]:
    """Resolve the FPM opt-in source while preserving the legacy port switch."""
    if os.environ.get("DYN_FORWARDPASS_METRIC_PORT"):
        return "DYN_FORWARDPASS_METRIC_PORT"
    if not dynamo_config.fpm_trace:
        return None

    unsupported_role = _unsupported_fpm_trace_role(dynamo_config)
    if unsupported_role is None:
        return "--fpm-trace/DYN_FPM_TRACE"

    logging.warning(
        "--fpm-trace/DYN_FPM_TRACE is enabled, but SGLang %s workers do not create a Dynamo "
        "FPM relay. Trace-based FPM activation is disabled for this worker.",
        unsupported_role,
    )
    return None


def use_modelexpress_remote_instance(args: Any) -> bool:
    return (
        getattr(args, "load_format", None) == "remote_instance"
        and getattr(args, "remote_instance_weight_loader_backend", None)
        == "modelexpress"
    )


def is_object_storage_path(model_path: str) -> bool:
    return model_path.startswith(("s3://", "gs://", "az://"))


def should_fetch_model(args: Any, model_path: str) -> bool:
    if os.path.exists(model_path):
        return False
    if is_object_storage_path(model_path):
        return False
    return not use_modelexpress_remote_instance(args)


# Register SGLang-specific encoders with the shared system
@register_encoder(Config)
def _preprocess_for_encode_config(
    config: Config,
) -> Dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
    """Convert Config object to dictionary for encoding."""
    return {
        "server_args": config.server_args,
        "dynamo_args": config.dynamo_args,
        "serving_mode": (
            config.serving_mode.value if config.serving_mode is not None else "None"
        ),
    }


def _validate_parser_flags(
    sglang_val: Optional[str], dynamo_val: Optional[str], name: str
) -> None:
    """Validate that --{name} (SGLang) and --dyn-{name} (Dynamo) are not both set."""
    if sglang_val and dynamo_val:
        logging.error(f"Cannot use both --{name} and --dyn-{name}.")
        sys.exit(1)


def _has_cli_flag(args: list[str], flag: str) -> bool:
    """Return True when a CLI flag is present in '--flag val' or '--flag=val' form."""
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in args)


def _get_last_cli_flag_value(args: list[str], flag: str) -> Optional[str]:
    """Return the last CLI flag value from '--flag val' or '--flag=val' form."""
    prefix = f"{flag}="
    value = None
    for idx, arg in enumerate(args):
        if arg.startswith(prefix):
            value = arg[len(prefix) :]
        if arg == flag:
            if idx + 1 >= len(args):
                continue
            value = args[idx + 1]
    return value


def _remove_cli_flag_and_value(args: list[str], flag: str) -> list[str]:
    """Remove a flag from CLI args, supporting '--flag val' and '--flag=val' forms."""
    updated: list[str] = []
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg == flag:
            skip_next = True
            continue
        if arg.startswith(f"{flag}="):
            continue
        updated.append(arg)
    return updated


def _set_cli_flag_value(args: list[str], flag: str, value: str) -> list[str]:
    """Set a flag value once, preserving argparse's last-value-wins behavior."""
    updated = _remove_cli_flag_and_value(args, flag)
    updated.extend([flag, value])
    return updated


def _normalize_multimodal_disaggregation_args(
    unknown: list[str], dynamo_config: "DynamoConfig"
) -> list[str]:
    """Map Dynamo's canonical multimodal args to SGLang's current flags."""
    disaggregation_mode = _get_last_cli_flag_value(unknown, "--disaggregation-mode")
    if disaggregation_mode is None:
        if dynamo_config.dedicated_mm_encoder and not dynamo_config.multimodal_worker:
            raise ValueError(
                "--dedicated-mm-encoder requires --disaggregation-mode=pd, "
                "--disaggregation-mode=prefill, or --disaggregation-mode=decode."
            )
        return unknown

    requested_disaggregation_mode = disaggregation_mode
    if disaggregation_mode in {
        DisaggregationMode.AGGREGATED.value,
        PREFILL_DECODE_DISAGGREGATION_MODE,
    }:
        unknown = _set_cli_flag_value(unknown, "--disaggregation-mode", "null")
        disaggregation_mode = "null"

    if disaggregation_mode == DisaggregationMode.ENCODE.value:
        if dynamo_config.dedicated_mm_encoder:
            raise ValueError(
                "--dedicated-mm-encoder is for PD/P/D workers that consume or "
                "forward embeddings from a separate encode worker. Do not "
                "combine it with --disaggregation-mode=encode."
            )
        if not dynamo_config.enable_multimodal:
            logging.warning(
                "--disaggregation-mode=encode is only valid for SGLang "
                "multimodal EPD; treating it as --enable-multimodal "
                "--disaggregation-mode=encode for this release."
            )
            dynamo_config.enable_multimodal = True
        dynamo_config.multimodal_encode_worker = True
        return _remove_cli_flag_and_value(unknown, "--disaggregation-mode")

    internal_multimodal_role = (
        requested_disaggregation_mode == PREFILL_DECODE_DISAGGREGATION_MODE
        or disaggregation_mode in {"prefill", "decode"}
    )
    if dynamo_config.dedicated_mm_encoder and not internal_multimodal_role:
        raise ValueError(
            "--dedicated-mm-encoder only applies to --disaggregation-mode=pd, "
            "--disaggregation-mode=prefill, or --disaggregation-mode=decode."
        )

    if (
        dynamo_config.enable_multimodal
        and dynamo_config.dedicated_mm_encoder
        and not dynamo_config.multimodal_encode_worker
        and not dynamo_config.multimodal_worker
        and internal_multimodal_role
    ):
        dynamo_config.multimodal_worker = True

    return unknown


def _load_disagg_config_section(config_path: str, config_key: str) -> dict[str, Any]:
    """
    Load a disaggregated config section from YAML.

    The selected section must exist and be a dictionary.
    """
    logging.info(f"Loading disagg config section '{config_key}' from {config_path}")

    path = Path(config_path)
    if not path.exists():
        raise ValueError(f"Disagg config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    if not isinstance(config_data, dict):
        raise ValueError(
            f"Disagg config file must contain a dictionary, got {type(config_data).__name__}"
        )

    available_keys = list(config_data.keys())
    if config_key not in config_data:
        raise ValueError(
            f"Disagg config key '{config_key}' not found in {config_path}. "
            f"Available keys: {available_keys}"
        )

    section_data = config_data[config_key]
    if not isinstance(section_data, dict):
        raise ValueError(
            f"Disagg config section '{config_key}' must be a dictionary, got {type(section_data).__name__}"
        )

    return section_data


def _dump_disagg_config_section(disagg_config: dict[str, Any]) -> str:
    """Dump the disaggregation configuration section to a YAML file."""
    temp_fd, temp_path = tempfile.mkstemp(suffix=".yaml", prefix="dynamo_config_")

    try:
        with os.fdopen(temp_fd, "w") as f:
            yaml.dump(disagg_config, f)
        logging.info("Successfully wrote config section to temp file")
    except Exception:
        os.unlink(temp_path)
        raise

    return temp_path


async def parse_args(args: list[str]) -> Config:
    """Parse CLI arguments and return combined configuration.
    Download the model if necessary.

    Args:
        args: Command-line argument strings.
    Returns:
        Config object with server_args and dynamo_args.

    Raises:
        SystemExit: If arguments are invalid or incompatible.
    """
    # Help must match the parser the worker will actually use. Diffusion
    # workers parse engine args with sglang's diffusion ServerArgs, so route
    # --help through that parser instead of letting the Dynamo/LLM parsers
    # print LLM engine options that a diffusion worker would reject. Checked
    # on the raw strings because argparse exits on -h in whichever parser
    # sees it first.
    if ("-h" in args or "--help" in args) and (
        "--image-diffusion-worker" in args or "--video-generation-worker" in args
    ):
        _print_diffusion_worker_help()
        sys.exit(0)

    runtime_argspec = DynamoRuntimeArgGroup()
    dynamo_sglang_argspec = DynamoSGLangArgGroup()

    parser = argparse.ArgumentParser(
        description="Dynamo SGLang worker configuration",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    runtime_argspec.add_arguments(parser)
    dynamo_sglang_argspec.add_arguments(parser)

    sglang_only_parser = argparse.ArgumentParser(add_help=False)
    ServerArgs.add_cli_args(sglang_only_parser)

    # Add "gms" to --load-format choices so it passes argparse validation.
    # The actual loader class is set in main.py when load_format == "gms".
    for action in sglang_only_parser._actions:
        if getattr(action, "dest", None) == "load_format" and action.choices:
            action.choices = list(action.choices) + ["gms"]
            break

    # trick to add sglang flags to a specific group without breaking the Dynamo groups.
    sg = parser.add_argument_group(
        "SGLang Engine Options. Please refer to SGLang documentation for more details."
    )
    for action in sglang_only_parser._actions:
        if not action.option_strings:
            continue
        sg._group_actions.append(action)

    # Router advertisement flags are parsed into their own config object rather
    # than flattened onto DynamoConfig: the router's --router-kv-events lands on
    # `use_kv_events`, which DynamoConfig already uses for "this worker
    # publishes KV events". Registered here for --help only; parsed below.
    register_worker_router_help(parser)

    dynamo_args, unknown = parser.parse_known_args(args)

    dynamo_config = DynamoConfig.from_cli_args(dynamo_args)
    # Consume the router flags before the SGLang parser sees the remainder.
    dynamo_config.router_advertisement, unknown = parse_worker_router_config(unknown)
    dynamo_config.validate()

    # Image/video diffusion workers configure DiffGenerator, whose options
    # live in SGLang's *diffusion* ServerArgs — a different dataclass from the
    # LLM ServerArgs everything below parses against. Branch off before any
    # LLM-specific processing so diffusion engine args are parsed natively.
    if dynamo_config.image_diffusion_worker or dynamo_config.video_generation_worker:
        return await _resolve_diffusion_worker_config(unknown, dynamo_config)

    # Dealing with SGLang native configs
    temp_config_file = None
    if dynamo_config.disagg_config and dynamo_config.disagg_config_key:
        section_data = _load_disagg_config_section(
            dynamo_config.disagg_config, dynamo_config.disagg_config_key
        )

        temp_config_file = _dump_disagg_config_section(section_data)

        # Remove any existing --config (both '--config val' and '--config=val' forms)
        unknown = _remove_cli_flag_and_value(unknown, "--config")
        unknown.append("--config")
        unknown.append(temp_config_file)

    try:
        if "--config" in unknown:
            config_merger = ConfigArgumentMerger(parser=sglang_only_parser)
            unknown = config_merger.merge_config_with_args(unknown)

        unknown = _normalize_multimodal_disaggregation_args(unknown, dynamo_config)
        dynamo_config.validate_multimodal_topology()

        parsed_args = sglang_only_parser.parse_args(unknown)
    finally:
        if temp_config_file and os.path.exists(temp_config_file):
            try:
                os.unlink(temp_config_file)
            except OSError as e:
                logging.warning(
                    "Failed to clean up temp config file %s: %s",
                    temp_config_file,
                    e,
                )

    bootstrap_port = _reserve_disaggregation_bootstrap_port()

    # Auto-set bootstrap port if not provided
    if not any(arg.startswith("--disaggregation-bootstrap-port") for arg in unknown):
        args_dict = vars(parsed_args)
        args_dict["disaggregation_bootstrap_port"] = bootstrap_port
        parsed_args = Namespace(**args_dict)

    # Dynamo argument processing
    # If an endpoint is provided, validate and use it
    # otherwise fall back to default endpoints
    namespace = dynamo_config.namespace

    # Dynamo's parser consumes --enable-multimodal; forward it to SGLang.
    if dynamo_config.enable_multimodal:
        parsed_args.enable_multimodal = True

    # If --embedding-worker is set, also set SGLang's --is-embedding flag
    if dynamo_config.embedding_worker:
        parsed_args.is_embedding = True

    # Enable encoder_only mode for multimodal encode workers to load only vision encoder
    # This significantly reduces memory usage by avoiding loading the full LLM weights
    if dynamo_config.multimodal_encode_worker:
        parsed_args.encoder_only = True

    endpoint = dynamo_config.endpoint
    if endpoint is None:
        if dynamo_config.embedding_worker:
            endpoint = f"dyn://{namespace}.backend.generate"
        elif dynamo_config.image_diffusion_worker:
            endpoint = f"dyn://{namespace}.backend.generate"
        elif dynamo_config.video_generation_worker:
            endpoint = f"dyn://{namespace}.backend.generate"
        elif (
            hasattr(parsed_args, "disaggregation_mode")
            and parsed_args.disaggregation_mode == "prefill"
        ):
            endpoint = f"dyn://{namespace}.prefill.generate"
        elif dynamo_config.multimodal_encode_worker:
            endpoint = f"dyn://{namespace}.encode.generate"
        elif (
            dynamo_config.multimodal_worker
            and parsed_args.disaggregation_mode == "prefill"
        ):
            endpoint = f"dyn://{namespace}.prefill.generate"
        else:
            endpoint = f"dyn://{namespace}.backend.generate"

    # Always parse the endpoint (whether auto-generated or user-provided)
    parsed_namespace, parsed_component_name, parsed_endpoint_name = parse_endpoint(
        endpoint
    )

    # Native and Dynamo tool parsers both construct tool calls, so they remain
    # mutually exclusive. Reasoning parsers intentionally may be paired: the
    # native parser gates guided decoding while Dynamo constructs the response.
    _validate_parser_flags(
        parsed_args.tool_call_parser,
        dynamo_config.dyn_tool_call_parser,
        "tool-call-parser",
    )

    if dynamo_config.custom_jinja_template and dynamo_config.use_sglang_tokenizer:
        logging.error(
            "Cannot use --custom-jinja-template and --use-sglang-tokenizer together. "
            "--custom-jinja-template requires Dynamo's preprocessor to apply the template, "
            "while --use-sglang-tokenizer bypasses Dynamo's preprocessor entirely."
            "If you want to use the SGLang tokenizer with a custom chat template, "
            "please use the --chat-template argument from SGLang."
        )
        sys.exit(1)

    # Replaces any environment variables or home dir (~) to get absolute path
    expanded_template_path = None
    if dynamo_config.custom_jinja_template:
        expanded_template_path = os.path.expandvars(
            os.path.expanduser(dynamo_config.custom_jinja_template)
        )
        # Validate custom Jinja template file exists
        if not os.path.isfile(expanded_template_path):
            raise FileNotFoundError(
                f"Custom Jinja template file not found: {expanded_template_path}"
            )

    model_path = parsed_args.model_path

    # --served-model-name may pack several names (whitespace-/comma-separated);
    # the first is the primary, the rest are aliases. Split BEFORE the
    # model_path fallback so a model path containing whitespace doesn't produce
    # spurious aliases.
    served_names = split_served_model_names(parsed_args.served_model_name)
    if served_names:
        parsed_args.served_model_name = served_names[0]
        dynamo_config.served_model_aliases = served_names[1:]
        if served_names[1:]:
            logging.info(
                "Multi-name registration: primary=%r, aliases=%s",
                served_names[0],
                served_names[1:],
            )

    # Name the model — falls back to model_path only if neither
    # --served-model-name nor an env var supplied one.
    if not parsed_args.served_model_name:
        parsed_args.served_model_name = model_path
    # Download the model if necessary using modelexpress.
    # We don't set `parsed_args.model_path` to the local path fetch_model returns
    # because sglang will send this to its pipeline-parallel workers, which may
    # not have the local path.
    # sglang will attempt to download the model again, but find it in the HF cache.
    # For non-HF models use a path instead of an HF name, and ensure all workers have
    # that path (ideally via a shared folder).
    if should_fetch_model(parsed_args, model_path):
        await fetch_model(model_path)

    if is_snapshot_enabled():
        configure_snapshot_capture_env()

    # TODO: sglang downloads the model in `from_cli_args`, which means we had to
    # fetch_model (download the model) here, in `parse_args`. `parse_args` should not
    # contain code to download a model, it should only parse the args.

    # ServerArgs is read-only after resolution, so apply Dynamo defaults first.
    fpm_source = _forward_pass_metrics_source(dynamo_config)
    if fpm_source and not getattr(parsed_args, "enable_forward_pass_metrics", False):
        parsed_args.enable_forward_pass_metrics = True
        logging.info("Enabled forward_pass_metrics from %s", fpm_source)

    if (
        parsed_args.dllm_algorithm
        and getattr(parsed_args, "max_running_requests", None) is None
    ):
        parsed_args.max_running_requests = 8
        logging.info("Defaulting max_running_requests to 8 for diffusion worker")

    # Dynamo expects disjoint output_ids; ServerArgs is read-only after resolution.
    parsed_args.incremental_streaming_output = True
    server_args = ServerArgs.from_cli_args(parsed_args)
    if server_args.get_model_config().is_multimodal:
        ensure_sglang_tensor_image_size()

    if getattr(server_args, "schedule_low_priority_values_first", False):
        raise ValueError(
            "--schedule-low-priority-values-first is not supported in Dynamo's "
            "SGLang integration. Dynamo normalizes request priority so higher "
            "values are always higher priority at the API layer."
        )

    if dynamo_config.use_sglang_tokenizer:
        warnings.warn(
            "--use-sglang-tokenizer is deprecated and will be removed in a future "
            "release. Use '--dyn-chat-processor sglang' on the frontend instead, "
            "which provides the same SGLang-native pre/post processing with KV "
            "router support.",
            FutureWarning,
            stacklevel=2,
        )
        logging.info("Using SGLang's built in tokenizer")
    else:
        logging.info("Using dynamo's built in tokenizer")

    # Derive use_kv_events from server_args.kv_events_config
    # Check that kv_events_config exists AND publisher is not "null" ("zmq" or any future publishers)
    use_kv_events = False
    if server_args.kv_events_config:
        try:
            kv_cfg = json.loads(server_args.kv_events_config)
            use_kv_events = kv_cfg.get("publisher", "null") != "null"
        except json.JSONDecodeError:
            logging.warning(
                f"Failed to parse kv_events_config: {server_args.kv_events_config}"
            )
    logging.info(
        f"Derived use_kv_events={use_kv_events} from kv_events_config={server_args.kv_events_config}"
    )

    # Auto-detect diffusion worker mode if dllm_algorithm
    diffusion_worker = server_args.dllm_algorithm is not None

    dynamo_config.namespace = parsed_namespace
    dynamo_config.component = parsed_component_name
    dynamo_config.endpoint = parsed_endpoint_name
    dynamo_config.custom_jinja_template = expanded_template_path
    dynamo_config.diffusion_worker = diffusion_worker
    dynamo_config.use_kv_events = use_kv_events

    logging.debug(f"Dynamo configs: {dynamo_config}")

    return Config(server_args, dynamo_config)


def _print_diffusion_worker_help() -> None:
    """Print combined Dynamo + native diffusion engine options."""
    dynamo_parser = argparse.ArgumentParser(
        prog="dynamo.sglang",
        description="Dynamo SGLang diffusion worker configuration",
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=False,
    )
    DynamoRuntimeArgGroup().add_arguments(dynamo_parser)
    DynamoSGLangArgGroup().add_arguments(dynamo_parser)
    print(dynamo_parser.format_help())
    print(
        "SGLang Diffusion Engine Options (native sglang diffusion ServerArgs;"
        " every option below is forwarded to the engine):\n"
    )
    diffusion_parser, _ = build_diffusion_parser()
    print(diffusion_parser.format_help())


async def _resolve_diffusion_worker_config(
    unknown: List[str], dynamo_config: "DynamoConfig"
) -> Config:
    """Resolve Config for image/video diffusion workers.

    Engine arguments are parsed with SGLang's native diffusion ServerArgs CLI
    (see diffusion_args.py), so every native diffusion engine argument is
    reachable — no hand-copied stub. The returned server_args is a
    DiffusionWorkerArgs adapter: engine fields resolve from the natively
    parsed diffusion ServerArgs, Dynamo-side settings live on the adapter.
    """
    worker_type = (
        "image diffusion"
        if dynamo_config.image_diffusion_worker
        else "video generation"
    )
    logging.info(
        f"{worker_type.title()} worker detected: parsing engine args with "
        "SGLang's native diffusion ServerArgs"
    )

    # Fetch the model before the full parse: sglang's diffusion ServerArgs
    # resolves model info during argument resolution, so the weights must be
    # available first (mirrors the LLM path, which fetches before
    # ServerArgs.from_cli_args).
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--model-path", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args(unknown)
    if not pre_args.model_path:
        raise ValueError("--model-path is required for diffusion workers")
    if should_fetch_model(argparse.Namespace(), pre_args.model_path):
        await fetch_model(pre_args.model_path)

    _parsed_args, server_args = parse_diffusion_args(unknown)

    # --served-model-name may pack several names; first is primary.
    served_names = split_served_model_names(server_args.served_model_name)
    if served_names:
        server_args.served_model_name = served_names[0]
        # If the engine args natively define served_model_name (newer sglang),
        # keep them consistent with the adapter: the engine must not receive
        # the unsplit multi-name string.
        if hasattr(server_args.engine_args, "served_model_name"):
            server_args.engine_args.served_model_name = served_names[0]
        dynamo_config.served_model_aliases = served_names[1:]
        if served_names[1:]:
            logging.info(
                "Multi-name registration: primary=%r, aliases=%s",
                served_names[0],
                served_names[1:],
            )

    if is_snapshot_enabled():
        configure_snapshot_capture_env()

    endpoint = (
        dynamo_config.endpoint or f"dyn://{dynamo_config.namespace}.backend.generate"
    )
    parsed_namespace, parsed_component_name, parsed_endpoint_name = parse_endpoint(
        endpoint
    )
    dynamo_config.namespace = parsed_namespace
    dynamo_config.component = parsed_component_name
    dynamo_config.endpoint = parsed_endpoint_name
    # dllm (text diffusion LLM) does not apply to image/video workers. The
    # old stub accidentally made this True (False is not None); it was never
    # load-bearing because main.py dispatches image/video workers first.
    dynamo_config.diffusion_worker = False
    dynamo_config.use_kv_events = False

    logging.debug(f"Dynamo configs: {dynamo_config}")

    return Config(server_args, dynamo_config)


@contextlib.contextmanager
def reserve_free_port(host: str = "localhost") -> Generator[int, None, None]:
    """Find and reserve a free port until context exits.

    Args:
        host: Host address to bind to.

    Yields:
        Available port number.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, 0))
        _, port = sock.getsockname()
        yield port
    finally:
        sock.close()


def _reserve_disaggregation_bootstrap_port() -> int:
    """Reserve a unique port for disaggregation bootstrap.

    Returns:
        Available port number.
    """
    with reserve_free_port() as port:
        return port
