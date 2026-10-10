# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source-backed protocol compatibility assessment and supporting generators."""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

from .assessment.workflow import assess, run_configured
from .common.paths import ROOT
from .generation import inventory, release_fixtures
from .inputs import pins


def assess_command(argv: list[str]) -> int:
    """Select explicit revisions or configured pins without separate CLI stacks."""
    parser = argparse.ArgumentParser(
        prog="python -m scripts.protocol_compatibility assess",
        description="Assess request contracts; static success does not establish runtime parity.",
    )
    parser.add_argument("--dynamo-repo", type=Path, default=ROOT)
    parser.add_argument("--dynamo-commit", required=True)
    parser.add_argument("--upstream-repo", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--platform", help="select the configured version pin")
    selection.add_argument(
        "--upstream-commit", help="explicit immutable source revision"
    )
    history = parser.add_mutually_exclusive_group()
    history.add_argument(
        "--baseline-dynamo", help="compare configured pins at this revision"
    )
    history.add_argument(
        "--previous", type=Path, help="previous direct-assessment report.json"
    )
    parser.add_argument(
        "--upstream-candidate", help="check without adopting configured pins"
    )
    parser.add_argument(
        "--crate-cache", type=Path, help="checksum-verified Cargo .crate archives"
    )
    parser.add_argument(
        "--pipeline", choices=("token", "text", "unspecified"), default="unspecified"
    )
    parser.add_argument("--decisions", type=Path)
    parser.add_argument("--support-policy", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.upstream_commit:
        if args.baseline_dynamo or args.upstream_candidate:
            parser.error(
                "--baseline-dynamo and --upstream-candidate require --platform; use --previous for explicit revisions"
            )
        return assess(args)
    return run_configured(args)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = {
        "assess": assess_command,
        "check-pins": pins.main,
        "generate-inventory": inventory.main,
        "generate-release-fixtures": release_fixtures.main,
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=commands, help="use COMMAND --help for its options"
    )
    if not argv or argv[0] not in commands:
        parser.parse_args(argv)
    try:
        return commands[argv[0]](argv[1:])
    except (
        OSError,
        SyntaxError,
        ValueError,
        KeyError,
        TypeError,
        yaml.YAMLError,
        subprocess.SubprocessError,
    ) as error:
        print(f"Protocol compatibility tool error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
