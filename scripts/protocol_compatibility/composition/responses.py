# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate a composed Dynamo document without running an assessment.

Despite the module name, this CLI fills both request and response placeholders
through composition.openai. It shares assess's manifest and writes
openapi.composed.json plus composition.yaml.
Use it for schema/Serde fidelity checks; use assess for framework comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .openai import compose

DIRECTORY = Path(__file__).resolve().parent


def response_manifest() -> dict:
    """Load the bundled request/response manifest; file and YAML errors propagate."""
    return yaml.safe_load((DIRECTORY / "async-openai-0.42.1.yaml").read_bytes())


def main() -> None:
    """Compose CLI-selected JSON and pinned OpenAI YAML into a new directory.

    Parsing, composition and I/O errors propagate. A failed save can leave partial
    output; source files are unchanged.
    """
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
