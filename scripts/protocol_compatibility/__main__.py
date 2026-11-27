# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run an offline Dynamo/framework assessment from local OpenAPI files.

This CLI parses arguments and translates assessment failures to exit code 2.
assessment.openapi_workflow owns the assessment and its output directory;
reporting.openapi provides a separate CLI to rebuild reports from saved evidence.
See README.md for commands, input requirements, and output formats.
"""

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
    """Run assess with argument tokens, or process arguments when argv is None.

    Return 0 for no differences/gaps, 1 for review findings, or 2 for assessment
    failures reported on stderr. argparse exits on help or invalid syntax.
    See assess for output writes and partial results after a failure.
    """
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
    except Exception as error:
        # CLI boundary only: exit 1 means a completed assessment, not a crash.
        print(
            f"Protocol compatibility tool error ({type(error).__name__}): {error}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
