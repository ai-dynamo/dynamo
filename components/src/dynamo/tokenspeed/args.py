# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Argument parsing for the TokenSpeed backend."""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import os
import sys
from typing import Any, Dict, Optional, Sequence

from dynamo.common.config_dump import register_encoder
from dynamo.common.configuration.groups.runtime_args import (
    DynamoRuntimeArgGroup,
    DynamoRuntimeConfig,
)
from dynamo.common.utils.runtime import parse_endpoint

DEFAULT_ENDPOINT_COMPONENT = "backend"
DEFAULT_ENDPOINT_NAME = "generate"


class Config(DynamoRuntimeConfig):
    component: str
    use_kv_events: bool = False

    model: str
    served_model_name: Optional[str] = None
    server_args: Any

    def validate(self) -> None:
        DynamoRuntimeConfig.validate(self)
        self.use_kv_events = False


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
    # Dynamo streams token deltas. TokenSpeed's raw-token streaming path needs
    # stream_output=True to avoid cumulative token ids.
    server_args.stream_output = True

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

    endpoint = (
        config.endpoint
        or f"dyn://{config.namespace}.{DEFAULT_ENDPOINT_COMPONENT}.{DEFAULT_ENDPOINT_NAME}"
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
