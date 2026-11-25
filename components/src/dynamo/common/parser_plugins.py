# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Rust parser plugins selected with ``--tool-parser-plugin``.

Worker argument parsers build the ``--dyn-tool-call-parser`` and
``--dyn-reasoning-parser`` choices from the parser registry, so plugins load
from a bootstrap pass over the command line before those choices exist.
"""

import argparse
import sys
from typing import Optional, Sequence, Union

from dynamo._core import load_parser_plugins

TOOL_PARSER_PLUGIN_FLAG = "--tool-parser-plugin"


def add_parser_plugin_argument(
    parser: Union[argparse.ArgumentParser, argparse._ArgumentGroup]
) -> None:
    """Register the repeatable ``--tool-parser-plugin PATH`` flag."""
    parser.add_argument(
        TOOL_PARSER_PLUGIN_FLAG,
        dest="tool_parser_plugin",
        action="append",
        default=[],
        metavar="PATH",
        help="Load a native Rust parser plugin (a shared library built against the "
        "dynamo-parsers-v2 plugin SDK) before parser names are validated. Repeat to "
        "load several. Each process that selects or validates the plugin's parser "
        "names needs the same library: pass it to both the frontend and the worker.",
    )


def load_parser_plugins_from_argv(argv: Optional[Sequence[str]] = None) -> list[str]:
    """Load the plugins named in ``argv`` (default ``sys.argv[1:]``).

    Returns the parser families the libraries registered. Exits with an error
    message when a library cannot be loaded or registered.
    """
    bootstrap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    add_parser_plugin_argument(bootstrap)
    args, _ = bootstrap.parse_known_args(sys.argv[1:] if argv is None else argv)
    if not args.tool_parser_plugin:
        return []
    try:
        return load_parser_plugins(args.tool_parser_plugin)
    except ValueError as exc:
        raise SystemExit(f"error: {TOOL_PARSER_PLUGIN_FLAG}: {exc}") from exc
