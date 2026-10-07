#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate the Python media protocol models from the Rust schemas.

The Rust media types write their schemas to
lib/llm/src/protocols/openai/media_schemas/ (see media_schemas.rs there). The
generate-media-protocols pre-commit hook runs this script when a schema
changes. To run it by hand:

    uvx pre-commit run generate-media-protocols --all-files

With --check, the script writes nothing and fails when a committed model is
stale.

The models carry the constraints of the Rust types: an enum becomes a
``Literal`` of its values, and a numeric bound becomes a field constraint. A
worker then rejects a value the frontend would reject, where the value is
produced.
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCHEMAS = REPO / "lib/llm/src/protocols/openai/media_schemas"
MODELS = REPO / "components/src/dynamo/common/protocols"
MODULES = ("audio", "image", "video")

# The copyright check compares the last year in this range with the year the
# file last changed. Extend the range when a regeneration lands in a new year.
HEADER = """\
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Generated from lib/llm/src/protocols/openai/media_schemas/{module}.json by
# scripts/generate_media_protocols.py. Do not edit."""

OPTIONS = (
    "--input-file-type=openapi",
    "--output-model-type=pydantic_v2.BaseModel",
    "--target-python-version=3.10",
    # An enum becomes a Literal of its values, not an Enum class.
    "--enum-field-as-literal=all",
    "--collapse-root-models",
    # A field with a serde default is not Optional.
    "--strict-nullable",
    "--use-field-description",
    "--use-schema-description",
    "--disable-timestamp",
    "--formatters=builtin",
)


def generate(out_dir: Path) -> None:
    for module in MODULES:
        subprocess.run(
            [
                "datamodel-codegen",
                f"--input={SCHEMAS / f'{module}.json'}",
                f"--output={out_dir / f'{module}_protocol.py'}",
                f"--custom-file-header={HEADER.format(module=module)}",
                *OPTIONS,
            ],
            check=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail when a committed model is stale, and write nothing",
    )
    if not parser.parse_args().check:
        generate(MODELS)
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        generate(Path(tmp))
        stale = [
            f"{module}_protocol.py"
            for module in MODULES
            if (Path(tmp) / f"{module}_protocol.py").read_text()
            != (MODELS / f"{module}_protocol.py").read_text()
        ]
    if stale:
        print(
            f"stale in {MODELS}: {', '.join(stale)}. Run "
            "`uvx pre-commit run generate-media-protocols --all-files`."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
