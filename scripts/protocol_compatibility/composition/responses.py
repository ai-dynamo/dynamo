# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compose request and response imports with the existing pinned OpenAI engine.

This entrypoint only completes schema imports; it does not compare providers or
claim behavioral compatibility. The request manifest remains the shared base.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .openai import compose

DIRECTORY = Path(__file__).resolve().parent


def response_manifest() -> dict:
    """Use the same reviewed manifest as request assessment, never a fork."""
    return yaml.safe_load((DIRECTORY / "async-openai-0.42.1.yaml").read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--openai", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = response_manifest()
    result = compose(
        json.loads(args.raw.read_text()),
        manifest,
        baseline_bytes=args.openai.read_bytes(),
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "openapi.composed.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    (args.output_dir / "composition.yaml").write_text(yaml.safe_dump(manifest))


if __name__ == "__main__":
    main()
