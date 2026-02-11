#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import re
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader


def flatten_context(context, prefix=""):
    flat = {}
    for key, value in context.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            # Check if this dict contains only scalar values (leaf node)
            if all(not isinstance(v, dict) for v in value.values()):
                # This is a leaf dict, add all its items
                for sub_key, sub_value in value.items():
                    flat[f"{full_key}.{sub_key}"] = sub_value
            else:
                # This dict has nested dicts, recurse
                flat.update(flatten_context(value, full_key))
        else:
            flat[full_key] = value
    return flat


def parse_args(context):
    parser = argparse.ArgumentParser(
        description="Renders dynamo Dockerfiles from templates"
    )

    parser.add_argument(
        "--framework",
        type=str,
        default="vllm",
        help="Dockerfile framework to use [dynamo, vllm, sglang, trtllm]",
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
        help="CUDA version to use. [12.9, 13.0]",
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

    flat_context = flatten_context(context)
    for key, value in flat_context.items():
        arg_name = f"--{key}"
        parser.add_argument(
            arg_name,
            type=str,
            default=None,
            help=f"Override context value: {key} (default: {value})",
        )

    args = parser.parse_args()
    return args


def apply_overrides(context, args, flat_context_keys):
    for key in flat_context_keys:
        # Normalize the key for attribute lookup: argparse converts hyphens to underscores
        lookup_key = key.replace("-", "_")
        arg_value = getattr(args, lookup_key, None)
        if arg_value is not None:
            # Navigate through nested dict and set the value using original key (preserves dots)
            keys = key.split(".")
            current = context
            for k in keys[:-1]:
                current = current[k]
            current[keys[-1]] = arg_value
    return context

def validate_args(args):
    valid_inputs = {
        "vllm": {"runtime", "dev", "local-dev", "framework", "wheel_builder", "base"},
        "trtllm": {"runtime", "dev", "local-dev", "framework", "wheel_builder", "base"},
        "sglang": {"runtime", "dev", "local-dev", "wheel_builder", "base"},
        "dynamo": {"runtime", "dev", "local-dev", "frontend", "wheel_builder", "base"},
    }

    if args.framework in valid_inputs:
        if args.target in valid_inputs[args.framework]:
            return
        raise ValueError(
            f"Invalid input combination: [framework={args.framework},target={args.target}]"
        )

    raise ValueError(
        f"Invalid input combination: [framework={args.framework},target={args.target}]"
    )
    return


def render(args, context, script_dir):
    env = Environment(
        loader=FileSystemLoader(script_dir), trim_blocks=False, lstrip_blocks=True
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
    script_dir = Path(__file__).parent
    with open(f"{script_dir}/context.yaml", "r") as f:
        context = yaml.safe_load(f)

    args = parse_args(context)
    flat_context = flatten_context(context)
    apply_overrides(context, args, flat_context.keys())
    validate_args(args)

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
