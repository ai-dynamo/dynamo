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
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Generator, Optional

import yaml
from sglang.srt.server_args import ServerArgs
from sglang.srt.server_args_config_parser import ConfigArgumentMerger

from dynamo.common.config_dump import register_encoder
from dynamo.common.configuration.groups import DynamoRuntimeConfig
from dynamo.common.configuration.groups.runtime_args import DynamoRuntimeArgGroup
from dynamo.common.constants import DisaggregationMode
from dynamo.common.utils.runtime import parse_endpoint
from dynamo.llm import fetch_model
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.sglang.backend_args import DynamoSGLangArgGroup, DynamoSGLangConfig

configure_dynamo_logging()


class DynamoConfig(DynamoRuntimeConfig, DynamoSGLangConfig):
    """Combined configuration container for SGLang server and Dynamo args."""

    component: str
    diffusion_worker: bool = False
    use_kv_events: bool = False

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


# Register SGLang-specific encoders with the shared system
@register_encoder(Config)
def _preprocess_for_encode_config(
    config: Config,
) -> Dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
    """Convert Config object to dictionary for encoding."""
    return {
        "server_args": config.server_args,
        "dynamo_args": config.dynamo_args,
        "serving_mode": config.serving_mode.value
        if config.serving_mode is not None
        else "None",
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

    dynamo_args, unknown = parser.parse_known_args(args)

    dynamo_config = DynamoConfig.from_cli_args(dynamo_args)
    dynamo_config.validate()

    # Dealing with SGLang native configs
    temp_config_file = None  # Track temp file for cleanup
    if dynamo_config.disagg_config and dynamo_config.disagg_config_key:
        section_data = _load_disagg_config_section(
            dynamo_config.disagg_config, dynamo_config.disagg_config_key
        )

        temp_config_file = _dump_disagg_config_section(section_data)

        # Remove any existing --config (both '--config val' and '--config=val' forms)
        unknown = _remove_cli_flag_and_value(unknown, "--config")
        unknown.append("--config")
        unknown.append(temp_config_file)

    # Handle SGLang --config file merge if present.
    if "--config" in unknown:
        # Merge config file arguments with CLI arguments.
        # ConfigArgumentMerger API changed after SGLang v0.5.7:
        # - New API (post-v0.5.7): accepts parser= for proper store_true detection
        # - Old API (v0.5.7 and earlier): only accepts boolean_actions=
        # We use inspect.signature to detect the API rather than version checking
        # since unreleased builds may have the new API while still reporting v0.5.7.
        # Related upstream issue: https://github.com/sgl-project/sglang/issues/16256
        # Upstream fix PR: https://github.com/sgl-project/sglang/pull/16638
        import inspect

        sig = inspect.signature(ConfigArgumentMerger.__init__)
        if "parser" in sig.parameters:
            config_merger = ConfigArgumentMerger(parser=sglang_only_parser)
        else:
            # Legacy path: extract store_true actions manually
            boolean_actions = [
                action.dest
                for action in sglang_only_parser._actions
                if isinstance(action, argparse._StoreTrueAction)
            ]
            config_merger = ConfigArgumentMerger(boolean_actions=boolean_actions)
        unknown = config_merger.merge_config_with_args(unknown)

    parsed_args = sglang_only_parser.parse_args(unknown)

    # Clean up temp file if created
    if temp_config_file and os.path.exists(temp_config_file):
        try:
            os.unlink(temp_config_file)
        except Exception:
            logging.warning(f"Failed to clean up temp config file: {temp_config_file}")

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

    # If --embedding-worker is set, also set SGLang's --is-embedding flag
    if dynamo_config.embedding_worker:
        parsed_args.is_embedding = True

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
        elif dynamo_config.multimodal_processor:
            endpoint = f"dyn://{namespace}.processor.generate"
        elif dynamo_config.multimodal_encode_worker:
            endpoint = f"dyn://{namespace}.encoder.generate"
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

    # Validate parser flags: error if both --{name} and --dyn-{name} are set.
    # --dyn-{name} choices are validated by argparse; --{name} by SGLang.
    _validate_parser_flags(
        parsed_args.tool_call_parser,
        dynamo_config.dyn_tool_call_parser,
        "tool-call-parser",
    )
    _validate_parser_flags(
        parsed_args.reasoning_parser,
        dynamo_config.dyn_reasoning_parser,
        "reasoning-parser",
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
    # Name the model
    if not parsed_args.served_model_name:
        parsed_args.served_model_name = model_path
    # Download the model if necessary using modelexpress.
    # We don't set `parsed_args.model_path` to the local path fetch_model returns
    # because sglang will send this to its pipeline-parallel workers, which may
    # not have the local path.
    # sglang will attempt to download the model again, but find it in the HF cache.
    # For non-HF models use a path instead of an HF name, and ensure all workers have
    # that path (ideally via a shared folder).
    if not os.path.exists(model_path):
        await fetch_model(model_path)

    # TODO: sglang downloads the model in `from_cli_args`, which means we had to
    # fetch_model (download the model) here, in `parse_args`. `parse_args` should not
    # contain code to download a model, it should only parse the args.

    # For diffusion/video workers, create a minimal dummy ServerArgs since diffusion
    # doesn't use transformer models or sglang Engine - it uses DiffGenerator directly
    image_diffusion_worker = dynamo_config.image_diffusion_worker
    video_generation_worker = dynamo_config.video_generation_worker

    if image_diffusion_worker or video_generation_worker:
        worker_type = (
            "image diffusion" if image_diffusion_worker else "video generation"
        )
        logging.info(
            f"{worker_type.title()} worker detected with model: {model_path}, creating minimal ServerArgs stub"
        )
        # Create a minimal ServerArgs-like object that bypasses model config loading
        # Diffusion/video workers don't actually use ServerArgs - they use DiffGenerator
        import types

        server_args = types.SimpleNamespace()
        # Copy over any attrs that might be needed, but avoid triggering __post_init__
        server_args.model_path = model_path
        server_args.served_model_name = parsed_args.served_model_name
        server_args.enable_metrics = getattr(parsed_args, "enable_metrics", False)
        server_args.log_level = getattr(parsed_args, "log_level", "info")
        server_args.skip_tokenizer_init = True
        server_args.kv_events_config = getattr(parsed_args, "kv_events_config", None)
        server_args.tp_size = getattr(parsed_args, "tp_size", 1)
        server_args.dp_size = getattr(parsed_args, "dp_size", 1)
        server_args.speculative_algorithm = None
        server_args.disaggregation_mode = None
        server_args.dllm_algorithm = False
        server_args.load_format = None
        logging.info(
            f"Created stub ServerArgs for {worker_type}: model_path={server_args.model_path}"
        )
    else:
        server_args = ServerArgs.from_cli_args(parsed_args)

    # Dynamo's streaming handlers expect disjoint output_ids from SGLang (only new
    # tokens since last output), not cumulative tokens. When stream_output=True,
    # SGLang sends disjoint segments which Dynamo passes through directly.
    # Force stream_output=True for optimal streaming performance.
    server_args.stream_output = True

    if dynamo_config.use_sglang_tokenizer:
        logging.info(
            "Using SGLang's built in tokenizer. Setting skip_tokenizer_init to False"
        )
        server_args.skip_tokenizer_init = False
    else:
        logging.info(
            "Using dynamo's built in tokenizer. Setting skip_tokenizer_init to True"
        )
        server_args.skip_tokenizer_init = True

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
