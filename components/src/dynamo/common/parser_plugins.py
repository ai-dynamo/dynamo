# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from collections.abc import Sequence

from dynamo._core import load_parser_plugins

FLAG = "--tool-parser-plugin"


def add_parser_plugin_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        FLAG,
        action="append",
        default=[],
        metavar="LIBRARY",
        help="Load a trusted native parser plugin from an absolute library path (repeatable).",
    )


def load_plugins_from_args(argv: Sequence[str] | None = None) -> None:
    # Worker argparse choices read the native catalog during parser construction.
    bootstrap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    add_parser_plugin_argument(bootstrap)
    args, _ = bootstrap.parse_known_args(argv)
    load_configured_plugins(bootstrap, args.tool_parser_plugin)


def load_configured_plugins(parser: argparse.ArgumentParser, paths: list[str]) -> None:
    if paths:
        try:
            load_parser_plugins(paths)
        except ValueError as error:
            parser.error(str(error))
