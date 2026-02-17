#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import re
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined


def parse_args():
    parser = argparse.ArgumentParser(
        description="Renders dynamo Dockerfiles from templates"
    )
    parser.add_argument(
        "--framework",
        type=str,
        default="vllm",
        choices=["dynamo", "vllm", "sglang", "trtllm"],
        help="Dockerfile framework to use",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="runtime",
        help="Dockerfile target to use. Non-exhaustive examples: [runtime, dev, local-dev]",
    )
    parser.add_argument(
        "--platform",
        type=str,
        default="amd64",
        help="Dockerfile platform to use. [amd64, arm64]",
    )
    parser.add_argument(
        "--cuda-version",
        type=str,
        default="12.9",
        choices=["12.9", "13.0", "13.1"],
        help="CUDA version to use. [12.9 or 13.0 for vllm and sglang, 13.1 for trtllm]",
    )
    parser.add_argument("--make-efa", action="store_true", help="Enable AWS EFA")
    parser.add_argument(
        "--output-short-filename",
        action="store_true",
        help="Output filename is rendered.Dockerfile instead of <framework>-<target>-cuda<cuda_version>-<arch>-rendered.Dockerfile",
    )
    parser.add_argument(
        "--show-result",
        action="store_true",
        help="Prints the rendered Dockerfile to stdout.",
    )
    args = parser.parse_args()
    return args


def validate_args(args):
    valid_inputs = {
        "vllm": {
            "target": [
                "runtime",
                "dev",
                "local-dev",
                "framework",
                "wheel_builder",
                "base",
            ],
            "cuda_version": ["12.9", "13.0"],
        },
        "trtllm": {
            "target": [
                "runtime",
                "dev",
                "local-dev",
                "framework",
                "wheel_builder",
                "base",
            ],
            "cuda_version": ["13.1"],
        },
        "sglang": {
            "target": [
                "runtime",
                "dev",
                "local-dev",
                "wheel_builder",
                "base",
            ],
            "cuda_version": ["12.9", "13.0"],
        },
        "dynamo": {
            "target": [
                "runtime",
                "dev",
                "local-dev",
                "frontend",
                "wheel_builder",
                "base",
            ],
            "cuda_version": ["12.9", "13.0"],
        },
    }

    if args.framework in valid_inputs:
        if (
            args.target in valid_inputs[args.framework]["target"]
            and args.cuda_version in valid_inputs[args.framework]["cuda_version"]
        ):
            return
        raise ValueError(
            f"Invalid input combination: [framework={args.framework},target={args.target},cuda_version={args.cuda_version}]"
        )

    raise ValueError(
        f"Invalid input combination: [framework={args.framework},target={args.target},cuda_version={args.cuda_version}]"
    )


def render(args, context, script_dir):
    env = Environment(
        loader=FileSystemLoader(script_dir),
        trim_blocks=False,
        lstrip_blocks=True,
        undefined=StrictUndefined,  # Raise an error if a variable in the template is not provided in the context
    )
    template = env.get_template("Dockerfile.template")
    rendered = template.render(
        context=context,
        framework=args.framework,
        target=args.target,
        platform=args.platform,
        cuda_version=args.cuda_version,
        make_efa=args.make_efa,
    )
    # Replace all instances of 3+ newlines with 2 newlines
    cleaned = re.sub(r"\n{3,}", "\n\n", rendered)

    if args.output_short_filename:
        filename = "rendered.Dockerfile"
    else:
        filename = f"{args.framework}-{args.target}-cuda{args.cuda_version}-{args.platform}-rendered.Dockerfile"

    with open(f"{script_dir}/{filename}", "w") as f:
        f.write(cleaned)

    if args.show_result:
        print("##############")
        print("# Dockerfile #")
        print("##############")
        print(cleaned)
        print("##############")

    print(f"INFO: Generated Dockerfile written to {script_dir}/{filename}")

    return


def main():
    args = parse_args()
    validate_args(args)
    script_dir = Path(__file__).parent
    with open(f"{script_dir}/context.yaml", "r") as f:
        context = yaml.safe_load(f)

    render(args, context, script_dir)

    if args.target == "local-dev":
        print(
            "INFO: Remember to add --build-arg values for USER_UID and USER_GID when building a local-dev image!"
        )
        print(
            "      Recommendation: --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g)"
        )


if __name__ == "__main__":
    main()
