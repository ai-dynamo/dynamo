#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render kvbm-cd.yaml.j2 into a deployable DGD manifest.

Usage:
    deploy/kvbm-cd/render.py --image nvcr.io/.../foo:tag         # writes kvbm-cd.yaml next to the template
    deploy/kvbm-cd/render.py --image ... --output -              # writes to stdout
    deploy/kvbm-cd/render.py --image ... --output deploy.yaml    # explicit output path
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jinja2

HERE = Path(__file__).resolve().parent
TEMPLATE_NAME = "kvbm-cd.yaml.j2"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--image", required=True, help="Container image ref for all four services"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-235B-A22B-FP8",
        help="HF model id; applied to --model + --served-model-name on prefill and decode (default: Qwen/Qwen3-235B-A22B-FP8)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=64,
        help="vLLM/KVBM block size; applied to KvbmHub, KvbmPrefill, and VllmWorker (default: 64)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=262144,
        help="Max sequence length; KvbmHub --max-seq-len + vLLM --max-model-len (default: 262144, sized for Qwen3-235B)",
    )
    parser.add_argument(
        "--output",
        default=str(HERE / "kvbm-cd.yaml"),
        help="Output path, or '-' for stdout (default: ./kvbm-cd.yaml)",
    )
    args = parser.parse_args()

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(HERE)),
        undefined=jinja2.StrictUndefined,
        keep_trailing_newline=True,
    )
    rendered = env.get_template(TEMPLATE_NAME).render(
        image=args.image,
        model=args.model,
        block_size=args.block_size,
        max_model_len=args.max_model_len,
    )

    if args.output == "-":
        sys.stdout.write(rendered)
    else:
        Path(args.output).write_text(rendered)
        print(f"wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
