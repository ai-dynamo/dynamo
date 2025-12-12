# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import contextlib
import logging
import os
import socket
import sys
import tempfile
from argparse import Namespace
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import yaml
from sglang.srt.server_args import ServerArgs
from sglang.srt.server_args_config_parser import ConfigArgumentMerger

from dynamo._core import get_reasoning_parser_names, get_tool_parser_names
from dynamo.common.config_dump import register_encoder
from dynamo.llm import fetch_llm
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.sglang import __version__

configure_dynamo_logging()

DYN_NAMESPACE = os.environ.get("DYN_NAMESPACE", "dynamo")
DEFAULT_ENDPOINT = f"dyn://{DYN_NAMESPACE}.backend.generate"

DYNAMO_ARGS: Dict[str, Dict[str, Any]] = {
    "endpoint": {
        "flags": ["--endpoint"],
        "type": str,
        "help": f"Dynamo endpoint string in 'dyn://namespace.component.endpoint' format. Example: {DEFAULT_ENDPOINT}",
    },
    "migration-limit": {
        "flags": ["--migration-limit"],
        "type": int,
        "default": 0,
        "help": "Maximum number of times a request may be migrated to a different engine worker",
    },
    "tool-call-parser": {
        "flags": ["--dyn-tool-call-parser"],
        "type": str,
        "default": None,
        "choices": get_tool_parser_names(),
        "help": "Tool call parser name for the model.",
    },
    "reasoning-parser": {
        "flags": ["--dyn-reasoning-parser"],
        "type": str,
        "default": None,
        "choices": get_reasoning_parser_names(),
        "help": "Reasoning parser name for the model. If not specified, no reasoning parsing is performed.",
    },
    "custom-jinja-template": {
        "flags": ["--custom-jinja-template"],
        "type": str,
        "default": None,
        "help": "Path to a custom Jinja template file to override the model's default chat template. This template will take precedence over any template found in the model repository. This template will be applied by Dynamo's preprocessor and cannot be used with --use-sglang-tokenizer.",
    },
    "endpoint-types": {
        "flags": ["--dyn-endpoint-types"],
        "type": str,
        "default": "chat,completions",
        "help": "Comma-separated list of endpoint types to enable. Options: 'chat', 'completions'. Default: 'chat,completions'. Use 'completions' for models without chat templates.",
    },
    "use-sglang-tokenizer": {
        "flags": ["--use-sglang-tokenizer"],
        "action": "store_true",
        "default": False,
        "help": "Use SGLang's tokenizer for pre and post processing. This bypasses Dynamo's preprocessor and only v1/chat/completions will be available through the Dynamo frontend. Cannot be used with --custom-jinja-template.",
    },
    "multimodal-processor": {
        "flags": ["--multimodal-processor"],
        "action": "store_true",
        "default": False,
        "help": "Run as multimodal processor component for handling multimodal requests",
    },
    "multimodal-encode-worker": {
        "flags": ["--multimodal-encode-worker"],
        "action": "store_true",
        "default": False,
        "help": "Run as multimodal encode worker component for processing images/videos",
    },
    "multimodal-worker": {
        "flags": ["--multimodal-worker"],
        "action": "store_true",
        "default": False,
        "help": "Run as multimodal worker component for LLM inference with multimodal data",
    },
    "embedding-worker": {
        "flags": ["--embedding-worker"],
        "action": "store_true",
        "default": False,
        "help": "Run as embedding worker component (Dynamo flag, also sets SGLang's --is-embedding)",
    },
    "dump-config-to": {
        "flags": ["--dump-config-to"],
        "type": str,
        "default": None,
        "help": "Dump debug config to the specified file path. If not specified, the config will be dumped to stdout at INFO level.",
    },
    "store-kv": {
        "flags": ["--store-kv"],
        "type": str,
        "choices": ["etcd", "file", "mem"],
        "default": os.environ.get("DYN_STORE_KV", "etcd"),
        "help": "Which key-value backend to use: etcd, mem, file. Etcd uses the ETCD_* env vars (e.g. ETCD_ENPOINTS) for connection details. File uses root dir from env var DYN_FILE_KV or defaults to $TMPDIR/dynamo_store_kv.",
    },
    "request-plane": {
        "flags": ["--request-plane"],
        "type": str,
        "choices": ["nats", "http", "tcp"],
        "default": os.environ.get("DYN_REQUEST_PLANE", "nats"),
        "help": "Determines how requests are distributed from routers to workers. 'tcp' is fastest [nats|http|tcp]",
    },
}


@dataclass
class DynamoArgs:
    namespace: str
    component: str
    endpoint: str
    migration_limit: int
    store_kv: str
    request_plane: str

    # tool and reasoning parser options
    tool_call_parser: Optional[str] = None
    reasoning_parser: Optional[str] = None
    custom_jinja_template: Optional[str] = None

    # endpoint types to enable
    dyn_endpoint_types: str = "chat,completions"

    # preprocessing options
    use_sglang_tokenizer: bool = False

    # multimodal options
    multimodal_processor: bool = False
    multimodal_encode_worker: bool = False
    multimodal_worker: bool = False

    # embedding options
    embedding_worker: bool = False
    # config dump options
    dump_config_to: Optional[str] = None


class DisaggregationMode(Enum):
    AGGREGATED = "agg"
    PREFILL = "prefill"
    DECODE = "decode"


class Config:
    """Combined configuration container for SGLang server and Dynamo args."""

    def __init__(self, server_args: ServerArgs, dynamo_args: DynamoArgs) -> None:
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
    return {
        "server_args": config.server_args,
        "dynamo_args": config.dynamo_args,
        "serving_mode": config.serving_mode.value
        if config.serving_mode is not None
        else "None",
    }


def _set_parser(
    sglang_str: Optional[str],
    dynamo_str: Optional[str],
    arg_name: str = "tool-call-parser",
) -> Optional[str]:
    """Resolve parser name from SGLang and Dynamo arguments.

    Args:
        sglang_str: Parser value from SGLang argument.
        dynamo_str: Parser value from Dynamo argument.
        arg_name: Name of the parser argument for logging.

    Returns:
        Resolved parser name, preferring Dynamo's value if both set.

    Raises:
        ValueError: If parser name is not valid.
    """
    # If both are present, give preference to dynamo_str
    if sglang_str is not None and dynamo_str is not None:
        logging.warning(
            f"--dyn-{arg_name} and --{arg_name} are both set. Giving preference to --dyn-{arg_name}"
        )
        return dynamo_str
    # If dynamo_str is not set, use try to use sglang_str if it matches with the allowed parsers
    elif sglang_str is not None:
        logging.warning(f"--dyn-{arg_name} is not set. Using --{arg_name}.")
        if arg_name == "tool-call-parser" and sglang_str not in get_tool_parser_names():
            raise ValueError(
                f"--{arg_name} is not a valid tool call parser. Valid parsers are: {get_tool_parser_names()}"
            )
        elif (
            arg_name == "reasoning-parser"
            and sglang_str not in get_reasoning_parser_names()
        ):
            raise ValueError(
                f"--{arg_name} is not a valid reasoning parser. Valid parsers are: {get_reasoning_parser_names()}"
            )
        return sglang_str
    else:
        return dynamo_str


def _extract_config_section(
    args: List[str], config_path: str, config_key: str
) -> tuple[List[str], str]:
    """
    Extract a section from nested YAML and create temp flat file.

    Args:
        args: CLI arguments list
        config_path: Path to the YAML config file
        config_key: Key to extract from nested YAML

    Returns:
        tuple: (modified args with temp file path, temp file path for cleanup)

    Raises:
        ValueError: If config file not found, key missing, or invalid format
    """
    logging.info(f"Extracting config section '{config_key}' from {config_path}")

    path = Path(config_path)
    if not path.exists():
        raise ValueError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    if not isinstance(config_data, dict):
        raise ValueError(
            f"Config file must contain a dictionary, got {type(config_data).__name__}"
        )

    available_keys = list(config_data.keys())
    logging.info(f"Available config keys in {config_path}: {available_keys}")

    if config_key not in config_data:
        raise ValueError(
            f"Config key '{config_key}' not found in {config_path}. "
            f"Available keys: {available_keys}"
        )

    section_data = config_data[config_key]

    if not isinstance(section_data, dict):
        raise ValueError(
            f"Config section '{config_key}' must be a dictionary, got {type(section_data).__name__}"
        )

    temp_fd, temp_path = tempfile.mkstemp(suffix=".yaml", prefix="dynamo_config_")

    try:
        with os.fdopen(temp_fd, "w") as f:
            yaml.dump(section_data, f)
        logging.info(f"Successfully wrote config section '{config_key}' to temp file")
    except Exception:
        os.unlink(temp_path)
        raise

    config_index = args.index("--config")
    args = list(args)
    args[config_index + 1] = temp_path

    return args, temp_path


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
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--version", action="version", version=f"Dynamo Backend SGLang {__version__}"
    )

    # Dynamo args
    for info in DYNAMO_ARGS.values():
        kwargs = {
            "default": info["default"] if "default" in info else None,
            "help": info["help"],
        }
        if "type" in info:
            kwargs["type"] = info["type"]
        if "choices" in info:
            kwargs["choices"] = info["choices"]
        if "action" in info:
            kwargs["action"] = info["action"]

        parser.add_argument(*info["flags"], **kwargs)

    # Config key argument (for nested configs)
    parser.add_argument(
        "--config-key",
        type=str,
        default=None,
        help="Key to select from nested config file (e.g., 'prefill', 'decode')",
    )

    # SGLang args
    bootstrap_port = _reserve_disaggregation_bootstrap_port()
    ServerArgs.add_cli_args(parser)

    # Handle config file if present
    temp_config_file = None  # Track temp file for cleanup
    if "--config" in args:
        # Check if --config-key is also present
        if "--config-key" in args:
            key_index = args.index("--config-key")
            config_key = args[key_index + 1]
            config_index = args.index("--config")
            config_path = args[config_index + 1]

            # Extract nested section to temp file
            args, temp_config_file = _extract_config_section(
                args, config_path, config_key
            )

            # Remove --config-key from args (not recognized by SGLang)
            args = args[:key_index] + args[key_index + 2 :]

        # Extract boolean actions from the parser to handle them correctly in YAML
        boolean_actions = []
        for action in parser._actions:
            if hasattr(action, "dest") and hasattr(action, "action"):
                if action.action in ["store_true", "store_false"]:
                    boolean_actions.append(action.dest)

        # Merge config file arguments with CLI arguments
        config_merger = ConfigArgumentMerger(boolean_actions=boolean_actions)
        args = config_merger.merge_config_with_args(args)

    parsed_args = parser.parse_args(args)

    # Clean up temp file if created
    if temp_config_file and os.path.exists(temp_config_file):
        try:
            os.unlink(temp_config_file)
        except Exception:
            logging.warning(f"Failed to clean up temp config file: {temp_config_file}")

    # Auto-set bootstrap port if not provided
    if not any(arg.startswith("--disaggregation-bootstrap-port") for arg in args):
        args_dict = vars(parsed_args)
        args_dict["disaggregation_bootstrap_port"] = bootstrap_port
        parsed_args = Namespace(**args_dict)

    # Dynamo argument processing
    # If an endpoint is provided, validate and use it
    # otherwise fall back to default endpoints
    namespace = os.environ.get("DYN_NAMESPACE", "dynamo")

    # If --embedding-worker is set, also set SGLang's --is-embedding flag
    if parsed_args.embedding_worker:
        parsed_args.is_embedding = True

    endpoint = parsed_args.endpoint
    if endpoint is None:
        if parsed_args.embedding_worker:
            endpoint = f"dyn://{namespace}.backend.generate"
        elif (
            hasattr(parsed_args, "disaggregation_mode")
            and parsed_args.disaggregation_mode == "prefill"
        ):
            endpoint = f"dyn://{namespace}.prefill.generate"
        elif parsed_args.multimodal_processor:
            endpoint = f"dyn://{namespace}.processor.generate"
        elif parsed_args.multimodal_encode_worker:
            endpoint = f"dyn://{namespace}.encoder.generate"
        elif (
            parsed_args.multimodal_worker
            and parsed_args.disaggregation_mode == "prefill"
        ):
            endpoint = f"dyn://{namespace}.prefill.generate"
        else:
            endpoint = f"dyn://{namespace}.backend.generate"

    # Always parse the endpoint (whether auto-generated or user-provided)
    endpoint_str = endpoint.replace("dyn://", "", 1)
    endpoint_parts = endpoint_str.split(".")
    if len(endpoint_parts) != 3:
        logging.error(
            f"Invalid endpoint format: '{endpoint}'. Expected 'dyn://namespace.component.endpoint' or 'namespace.component.endpoint'."
        )
        sys.exit(1)

    parsed_namespace, parsed_component_name, parsed_endpoint_name = endpoint_parts

    tool_call_parser = _set_parser(
        parsed_args.tool_call_parser,
        parsed_args.dyn_tool_call_parser,
        "tool-call-parser",
    )
    reasoning_parser = _set_parser(
        parsed_args.reasoning_parser,
        parsed_args.dyn_reasoning_parser,
        "reasoning-parser",
    )

    if parsed_args.custom_jinja_template and parsed_args.use_sglang_tokenizer:
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
    if parsed_args.custom_jinja_template:
        expanded_template_path = os.path.expandvars(
            os.path.expanduser(parsed_args.custom_jinja_template)
        )
        # Validate custom Jinja template file exists
        if not os.path.isfile(expanded_template_path):
            raise FileNotFoundError(
                f"Custom Jinja template file not found: {expanded_template_path}"
            )

    dynamo_args = DynamoArgs(
        namespace=parsed_namespace,
        component=parsed_component_name,
        endpoint=parsed_endpoint_name,
        migration_limit=parsed_args.migration_limit,
        store_kv=parsed_args.store_kv,
        request_plane=parsed_args.request_plane,
        tool_call_parser=tool_call_parser,
        reasoning_parser=reasoning_parser,
        custom_jinja_template=expanded_template_path,
        dyn_endpoint_types=parsed_args.dyn_endpoint_types,
        use_sglang_tokenizer=parsed_args.use_sglang_tokenizer,
        multimodal_processor=parsed_args.multimodal_processor,
        multimodal_encode_worker=parsed_args.multimodal_encode_worker,
        multimodal_worker=parsed_args.multimodal_worker,
        embedding_worker=parsed_args.embedding_worker,
        dump_config_to=parsed_args.dump_config_to,
    )
    logging.debug(f"Dynamo args: {dynamo_args}")

    # TODO: sglang downloads the model in `from_cli_args`, so we need to do it here.
    # That's unfortunate because `parse_args` isn't the right place for this. Fix.
    model_path = parsed_args.model_path
    if not parsed_args.served_model_name:
        parsed_args.served_model_name = model_path
    if not os.path.exists(model_path):
        parsed_args.model_path = await fetch_llm(model_path)

    server_args = ServerArgs.from_cli_args(parsed_args)

    if parsed_args.use_sglang_tokenizer:
        logging.info(
            "Using SGLang's built in tokenizer. Setting skip_tokenizer_init to False"
        )
        server_args.skip_tokenizer_init = False
    else:
        logging.info(
            "Using dynamo's built in tokenizer. Setting skip_tokenizer_init to True"
        )
        server_args.skip_tokenizer_init = True

    return Config(server_args, dynamo_args)


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


def parse_endpoint(endpoint: str) -> List[str]:
    """Parse endpoint string into namespace, component, and endpoint parts.

    Args:
        endpoint: Endpoint string in 'dyn://namespace.component.endpoint' format.

    Returns:
        List of [namespace, component, endpoint] strings.

    Raises:
        ValueError: If endpoint format is invalid.
    """
    endpoint_str = endpoint.replace("dyn://", "", 1)
    endpoint_parts = endpoint_str.split(".")
    if len(endpoint_parts) != 3:
        error_msg = (
            f"Invalid endpoint format: '{endpoint}'. "
            f"Expected 'dyn://namespace.component.endpoint' or 'namespace.component.endpoint'."
        )
        logging.error(error_msg)
        raise ValueError(error_msg)

    return endpoint_parts


def _reserve_disaggregation_bootstrap_port() -> int:
    """Reserve a unique port for disaggregation bootstrap.

    Returns:
        Available port number.
    """
    with reserve_free_port() as port:
        return port
