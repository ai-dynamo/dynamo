# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Argument parsing for the TokenSpeed backend."""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

from dynamo.common.config_dump import register_encoder
from dynamo.common.configuration.groups.runtime_args import (
    DynamoRuntimeArgGroup,
    DynamoRuntimeConfig,
)
from dynamo.common.constants import DisaggregationMode
from dynamo.common.utils.runtime import parse_endpoint
from dynamo.tokenspeed.disagg import (
    resolve_disaggregation_mode,
    validate_disagg_compatibility,
)

DEFAULT_ENDPOINT_COMPONENT = "backend"
DEFAULT_ENDPOINT_NAME = "generate"


class Config(DynamoRuntimeConfig):
    component: str
    disaggregation_mode: DisaggregationMode = DisaggregationMode.AGGREGATED
    use_kv_events: bool = False

    model: str
    served_model_name: Optional[str] = None
    server_args: Any
    # The TokenSpeed command line as given, which the worker hands to the
    # headless engine launcher (with the frontend address appended).
    engine_argv: List[str]
    # TokenSpeed's KV prefix granularity in tokens. Each model's cache recipe
    # picks it at runtime and the engine reports it only in its handshake,
    # which tokenspeed_engine does not surface to Python, so KV-event routing
    # needs it stated.
    kv_block_size: Optional[int] = None

    def validate(self) -> None:
        DynamoRuntimeConfig.validate(self)
        self.disaggregation_mode = resolve_disaggregation_mode(self.server_args)
        validate_disagg_compatibility(self.disaggregation_mode, self.server_args)
        if self.kv_block_size is not None and self.kv_block_size <= 0:
            raise ValueError("--kv-block-size must be positive")


@register_encoder(Config)
def _preprocess_for_encode_config(config: Config) -> Dict[str, Any]:
    data = dict(config.__dict__)
    server_args = data.get("server_args")
    if dataclasses.is_dataclass(server_args) and not isinstance(server_args, type):
        data["server_args"] = dataclasses.asdict(server_args)
    return data


def parse_args(argv: Optional[Sequence[str]] = None) -> Config:
    cli_args = list(argv) if argv is not None else sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Dynamo TokenSpeed worker configuration",
        formatter_class=argparse.RawTextHelpFormatter,
        allow_abbrev=False,
    )
    DynamoRuntimeArgGroup().add_arguments(parser)
    parser.add_argument(
        "--kv-block-size",
        type=int,
        default=None,
        help="TokenSpeed's KV prefix granularity in tokens, as the engine logs it "
        "at startup (prefix_granularity=N). Required for KV-event routing: "
        "--kv-events-config with enable_kv_cache_events. Must match the engine, "
        "or the router drops its KV events.",
    )

    server_args_cls = _server_args_cls()
    tokenspeed_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    server_args_cls.add_cli_args(tokenspeed_parser)

    # Keep TokenSpeed flags visible in --help while letting the dedicated
    # TokenSpeed parser own them, including its positional model argument.
    tokenspeed_group = parser.add_argument_group(
        "TokenSpeed Engine Options. Please refer to TokenSpeed documentation for more details."
    )
    for action in tokenspeed_parser._actions:
        if action.option_strings or action.dest == "model_path":
            tokenspeed_group._group_actions.append(action)

    dynamo_args, remaining = parser.parse_known_args(cli_args)
    config = Config.from_cli_args(dynamo_args)

    server_args = server_args_cls.from_cli_args(tokenspeed_parser.parse_args(remaining))
    config.engine_argv = list(remaining)

    config.model = server_args.model
    config.served_model_name = server_args.served_model_name or server_args.model
    server_args.served_model_name = config.served_model_name
    config.server_args = server_args

    config.validate()

    if config.custom_jinja_template:
        expanded_template_path = os.path.expanduser(
            os.path.expandvars(config.custom_jinja_template)
        )
        if not os.path.isfile(expanded_template_path):
            raise FileNotFoundError(
                f"Custom Jinja template file not found: {expanded_template_path}"
            )
        config.custom_jinja_template = expanded_template_path
    else:
        config.custom_jinja_template = None

    component = (
        "prefill"
        if config.disaggregation_mode == DisaggregationMode.PREFILL
        else DEFAULT_ENDPOINT_COMPONENT
    )
    endpoint = (
        config.endpoint
        or f"dyn://{config.namespace}.{component}.{DEFAULT_ENDPOINT_NAME}"
    )
    parsed_namespace, parsed_component_name, parsed_endpoint_name = parse_endpoint(
        endpoint
    )
    config.namespace = parsed_namespace
    config.component = parsed_component_name
    config.endpoint = parsed_endpoint_name

    return config


def _server_args_cls():
    module = importlib.import_module("tokenspeed.runtime.utils.server_args")
    return module.ServerArgs
