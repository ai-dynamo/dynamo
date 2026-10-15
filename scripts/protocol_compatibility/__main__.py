# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare Dynamo/framework request and response contracts from local OpenAPI files."""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml
from jsonpatch import JsonPatchException
from jsonpointer import JsonPointerException
from openapi_spec_validator.validation.exceptions import OpenAPIValidationError

from .assessment.openapi_workflow import DEFAULT_MANIFEST, assess


def main(argv=None) -> int:
    """Assess local specifications; return 2 for input/tool failures reported on stderr."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    assessment = commands.add_parser(
        "assess",
        help="Compose Dynamo and compare request/response contracts to a framework specification",
    )
    assessment.add_argument(
        "--dynamo",
        type=Path,
        required=True,
        help="Local Dynamo OpenAPI JSON/YAML file",
    )
    assessment.add_argument(
        "--framework",
        type=Path,
        required=True,
        help="Local framework OpenAPI JSON/YAML file",
    )
    assessment.add_argument(
        "--framework-name",
        required=True,
        help="Caller-selected lowercase identity (not dynamo)",
    )
    for side in ("dynamo", "framework"):
        assessment.add_argument(
            f"--{side}-metadata",
            type=Path,
            help=f"Optional JSON annotations for the {side} file, retained without verification",
        )
    assessment.add_argument(
        "--openai", type=Path, required=True, help="pinned async-openai openapi.yaml"
    )
    assessment.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    assessment.add_argument(
        "--response-adapters",
        type=Path,
        help="Optional hash-qualified SSE interpretations and explicit completion path mappings",
    )
    assessment.add_argument(
        "--oasdiff", type=Path, required=True, help="pinned oasdiff 1.33.0 executable"
    )
    assessment.add_argument("--output-dir", type=Path, required=True)
    assessment.set_defaults(run=assess)
    args = parser.parse_args(argv)
    try:
        return args.run(args)
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        yaml.YAMLError,
        subprocess.SubprocessError,
        JsonPatchException,
        JsonPointerException,
        OpenAPIValidationError,
    ) as error:
        print(f"Protocol compatibility tool error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
