# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import contextlib
import logging
import os
import socket
import sys
from argparse import Namespace
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


DYN_NAMESPACE = os.environ.get("DYN_NAMESPACE", "dynamo")
DEFAULT_ENDPOINT = f"dyn://{DYN_NAMESPACE}.backend.generate"


class DisaggregationMode(Enum):
    AGGREGATED = "agg"
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass
class DynamoArgs:
    """Dynamo-specific arguments for multimodal SGLang components"""

    namespace: str
    component: str
    endpoint: str
    downstream_endpoint: Optional[str] = None


class Config:
    """Config class following SGLang backend pattern"""

    def __init__(self, server_args: ServerArgs, dynamo_args: DynamoArgs):
        self.server_args = server_args
        self.dynamo_args = dynamo_args
        self.serving_mode = self._set_serving_strategy()

        # Convenient access to common attributes
        self.model = server_args.model_path
        self.served_model_name = getattr(server_args, "served_model_name", None)
        self.block_size = getattr(server_args, "kv_cache_block_size", None)
        self.namespace = dynamo_args.namespace
        self.component = dynamo_args.component
        self.endpoint = dynamo_args.endpoint
        self.downstream_endpoint = dynamo_args.downstream_endpoint

    def _set_serving_strategy(self):
        """Set serving mode based on disaggregation_mode"""
        if self.server_args.disaggregation_mode == "null":
            return DisaggregationMode.AGGREGATED
        elif self.server_args.disaggregation_mode == "prefill":
            return DisaggregationMode.PREFILL
        elif self.server_args.disaggregation_mode == "decode":
            return DisaggregationMode.DECODE
        else:
            return DisaggregationMode.AGGREGATED


def parse_args(component: str, args_list: Optional[List[str]] = None) -> Config:
    """
    Parse arguments following SGLang backend pattern with multimodal-specific defaults

    Args:
        component: The component type (processor, worker, encoder, prefill)
        args_list: Optional list of arguments to parse, defaults to sys.argv[1:]

    Returns:
        Config: Configuration object with server_args and dynamo_args
    """
    if args_list is None:
        args_list = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description=f"SGLang {component.title()} for Dynamo"
    )

    # Add Dynamo-specific arguments only (following SGLang backend pattern)
    parser.add_argument(
        "--endpoint",
        type=str,
        help=f"Dynamo endpoint string in 'dyn://namespace.component.endpoint' format. Example: {DEFAULT_ENDPOINT}",
    )
    parser.add_argument(
        "--downstream-endpoint",
        type=str,
        help="Downstream endpoint for pipeline communication",
    )

    # SGLang args - automatically add ALL SGLang arguments using ServerArgs.add_cli_args
    bootstrap_port = _reserve_disaggregation_bootstrap_port()
    ServerArgs.add_cli_args(parser)

    parsed_args = parser.parse_args(args_list)

    # Auto-set bootstrap port if not provided (following SGLang backend)
    if not any(arg.startswith("--disaggregation-bootstrap-port") for arg in args_list):
        args_dict = vars(parsed_args)
        args_dict["disaggregation_bootstrap_port"] = bootstrap_port
        parsed_args = Namespace(**args_dict)

    # Dynamo argument processing - determine endpoints
    endpoint = parsed_args.endpoint
    if endpoint is None:
        # Set default endpoints based on component and disaggregation mode
        if (
            hasattr(parsed_args, "disaggregation_mode")
            and parsed_args.disaggregation_mode == "prefill"
        ):
            endpoint = f"dyn://{DYN_NAMESPACE}.prefill.generate"
        elif (
            hasattr(parsed_args, "disaggregation_mode")
            and parsed_args.disaggregation_mode == "decode"
        ):
            endpoint = f"dyn://{DYN_NAMESPACE}.llm.generate"
        else:
            # Default endpoints for different components
            component_endpoint_map = {
                "processor": f"dyn://{DYN_NAMESPACE}.processor.generate",
                "encoder": f"dyn://{DYN_NAMESPACE}.encoder.generate",
                "worker": f"dyn://{DYN_NAMESPACE}.llm.generate",
                "prefill": f"dyn://{DYN_NAMESPACE}.prefill.generate",
            }
            endpoint = component_endpoint_map.get(
                component, f"dyn://{DYN_NAMESPACE}.{component}.generate"
            )

    # Set downstream endpoint if not provided
    downstream_endpoint = parsed_args.downstream_endpoint
    if downstream_endpoint is None:
        downstream_endpoint_map = {
            "processor": f"dyn://{DYN_NAMESPACE}.encoder.generate",  # processor -> encoder
            "encoder": f"dyn://{DYN_NAMESPACE}.llm.generate",  # encoder -> pd worker
            "worker": f"dyn://{DYN_NAMESPACE}.prefill.generate",  # pd worker -> prefill worker
            "prefill": None,  # end of pipeline
        }
        downstream_endpoint = downstream_endpoint_map.get(component)

    # Parse endpoint
    endpoint_str = endpoint.replace("dyn://", "", 1)
    endpoint_parts = endpoint_str.split(".")
    if len(endpoint_parts) != 3:
        logging.error(
            f"Invalid endpoint format: '{endpoint}'. Expected 'dyn://namespace.component.endpoint' or 'namespace.component.endpoint'."
        )
        raise ValueError(
            f"Invalid endpoint format: '{endpoint}'. Expected 'dyn://namespace.component.endpoint' or 'namespace.component.endpoint'."
        )

    parsed_namespace, parsed_component_name, parsed_endpoint_name = endpoint_parts

    dynamo_args = DynamoArgs(
        namespace=parsed_namespace,
        component=parsed_component_name,
        endpoint=parsed_endpoint_name,
        downstream_endpoint=downstream_endpoint,
    )

    # Create ServerArgs using SGLang's method (following backend pattern)
    server_args = ServerArgs.from_cli_args(parsed_args)

    # Auto-configure skip_tokenizer_init for frontend usage (following backend pattern)
    if not server_args.skip_tokenizer_init:
        logging.info(
            "When using multimodal processor, we handle tokenization in the pipeline. "
            "Automatically setting --skip-tokenizer-init to True for worker components."
        )
        if component in ["worker", "llm"]:
            server_args.skip_tokenizer_init = True

    return Config(server_args, dynamo_args)


def parse_endpoint(endpoint: str) -> List[str]:
    endpoint_str = endpoint.replace("dyn://", "", 1)
    endpoint_parts = endpoint_str.split(".")
    if len(endpoint_parts) != 3:
        logger.error(
            f"Invalid endpoint format: '{endpoint}'. Expected 'dyn://namespace.component.endpoint' or 'namespace.component.endpoint'."
        )
        sys.exit(1)

    return endpoint_parts


@contextlib.contextmanager
def reserve_free_port(host: str = "localhost"):
    """
    Find and reserve a free port until context exits.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, 0))
        _, port = sock.getsockname()
        yield port
    finally:
        sock.close()


def _reserve_disaggregation_bootstrap_port():
    """
    Each worker requires a unique port for disaggregation_bootstrap_port.
    We use an existing utility function that reserves a free port on your
    machine to avoid collisions.
    """
    with reserve_free_port() as port:
        return port
