#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve the base image for a given framework/target/cuda from context.yaml.

Prints the resolved image URI to stdout so it can be captured in shell scripts.

Usage:
    python resolve_base_image.py --framework vllm --cuda-version 12.9
    python resolve_base_image.py --framework dynamo --target frontend
    python resolve_base_image.py --framework sglang --cuda-version 13.0
"""

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve base image URI from container/context.yaml"
    )
    parser.add_argument(
        "--framework",
        required=True,
        choices=["vllm", "sglang", "trtllm", "dynamo"],
        help="Framework name",
    )
    parser.add_argument(
        "--target",
        default="runtime",
        choices=["runtime", "frontend", "planner"],
        help="Build target (default: runtime)",
    )
    parser.add_argument(
        "--cuda-version",
        help="CUDA version (e.g. 12.9, 13.0, 13.1) — required for runtime targets",
    )
    parser.add_argument(
        "--context-yaml",
        default=str(_REPO_ROOT / "container" / "context.yaml"),
        help="Path to context.yaml (default: container/context.yaml in repo root)",
    )
    args = parser.parse_args()

    try:
        import yaml
    except ImportError:
        print(
            "ERROR: pyyaml is required — install with: pip install pyyaml",
            file=sys.stderr,
        )
        sys.exit(1)

    context_yaml = Path(args.context_yaml)
    if not context_yaml.is_file():
        print(f"ERROR: context.yaml not found at {context_yaml}", file=sys.stderr)
        sys.exit(1)

    with open(context_yaml) as f:
        ctx = yaml.safe_load(f)

    if args.target == "frontend":
        image = ctx.get("dynamo", {}).get("frontend_image")
        if not image:
            print(
                "ERROR: frontend_image not found in context.yaml dynamo section",
                file=sys.stderr,
            )
            sys.exit(1)
        print(image)
        return

    if args.target == "planner":
        if args.framework != "dynamo":
            print(
                "ERROR: --target planner is only supported for --framework dynamo",
                file=sys.stderr,
            )
            sys.exit(1)
        planner_cfg = ctx.get("dynamo", {})
        runtime_image = planner_cfg.get("planner_runtime_image")
        runtime_image_tag = planner_cfg.get("planner_runtime_image_tag")
        if not runtime_image or not runtime_image_tag:
            print(
                "ERROR: planner_runtime_image/planner_runtime_image_tag not found in context.yaml dynamo section",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"{runtime_image}:{runtime_image_tag}")
        return

    # Runtime target
    if not args.cuda_version:
        print("ERROR: --cuda-version is required for runtime targets", file=sys.stderr)
        sys.exit(1)

    fw_config = ctx.get(args.framework, {})
    cuda_key = f"cuda{args.cuda_version}"
    cuda_config = fw_config.get(cuda_key, {})

    runtime_image = cuda_config.get("runtime_image")
    runtime_image_tag = cuda_config.get("runtime_image_tag")

    if not runtime_image or not runtime_image_tag:
        print(
            f"ERROR: Could not resolve base image for framework={args.framework} "
            f"cuda={args.cuda_version}. Keys runtime_image/runtime_image_tag not found "
            f"under {args.framework}.{cuda_key} in context.yaml",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"{runtime_image}:{runtime_image_tag}")


if __name__ == "__main__":
    main()
