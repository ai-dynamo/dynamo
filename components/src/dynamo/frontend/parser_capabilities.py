# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json

from dynamo._core import get_parser_capabilities
from dynamo.common.parser_plugins import (
    add_parser_plugin_argument,
    load_configured_plugins,
)


def print_parser_capabilities() -> None:
    print(json.dumps(get_parser_capabilities(), indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="List available Dynamo parsers")
    parser.add_argument("--list-parsers", action="store_true", required=True)
    add_parser_plugin_argument(parser)
    args = parser.parse_args()
    load_configured_plugins(parser, args.tool_parser_plugin)
    print_parser_capabilities()
