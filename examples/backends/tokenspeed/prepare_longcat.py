# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Add LongCat-Flash's declared model type to checkpoints that omit it."""

import argparse
import json
from pathlib import Path


def prepare(model_dir: Path) -> None:
    """Add the missing model type while preserving the original checkpoint config."""
    path = model_dir / "config.json"
    original = path.read_text()
    config = json.loads(original)
    if config.get("architectures") != ["LongcatFlashForCausalLM"]:
        raise ValueError("Expected a LongCat-Flash checkpoint")
    model_type = config.get("model_type")
    if model_type == "longcat_flash":
        print(f"LongCat metadata already prepared: {path}")
        return
    if model_type is not None:
        raise ValueError(f"Unexpected LongCat model_type: {model_type!r}")
    # The pinned model owner's LongcatFlashConfig declares this value.
    # Dynamo's frontend reads config.json directly instead of executing that class.
    backup = model_dir / "config.json.dynamo-original"
    if not backup.exists():
        backup.write_text(original)
    config["model_type"] = "longcat_flash"
    path.write_text(json.dumps(config, indent=2) + "\n")
    print(f"Prepared LongCat metadata: {path}; original: {backup}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    prepare(parser.parse_args().model_dir)
