# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Acquire OpenAPI documents and assess Dynamo/framework request contracts."""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml
from jsonpatch import JsonPatchException
from jsonpointer import JsonPointerException
from openapi_spec_validator.validation.exceptions import OpenAPIValidationError

from .acquisition.http import acquire
from .assessment.openapi_workflow import DEFAULT_MANIFEST, assess


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser(
        "acquire", help="Save an OpenAPI file or HTTP(S) URL with its checksum"
    )
    capture.add_argument(
        "--server",
        required=True,
        help="Server identity: dynamo or a lowercase framework slug",
    )
    source = capture.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="Published or live HTTP(S) OpenAPI URL")
    source.add_argument("--file", type=Path, help="Local JSON or YAML OpenAPI file")
    capture.add_argument(
        "--metadata",
        type=Path,
        help="Optional caller-supplied JSON annotations, retained without verification",
    )
    capture.add_argument("--output-dir", type=Path, required=True)
    capture.set_defaults(run=acquire)
    assessment = commands.add_parser(
        "assess", help="Compose Dynamo and compare its requests to a framework server"
    )
    assessment.add_argument(
        "--dynamo",
        type=Path,
        required=True,
        help="Dynamo JSON/YAML file or capture directory",
    )
    assessment.add_argument(
        "--framework",
        type=Path,
        required=True,
        help="Framework JSON/YAML file or capture directory",
    )
    assessment.add_argument(
        "--framework-name",
        required=True,
        help="Caller-selected identity; must match a capture's --server (not dynamo)",
    )
    assessment.add_argument(
        "--openai", type=Path, required=True, help="pinned async-openai openapi.yaml"
    )
    assessment.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
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
