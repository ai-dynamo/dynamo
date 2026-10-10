#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Resolve the one supported sanitized imageID attestation for a campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


class AttestationError(ValueError):
    """The campaign inventories cannot be mapped to one safe attestation."""


def _read_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AttestationError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise AttestationError(f"{path} is not a JSON object")
    return value


def _read_array(path: Path) -> list:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AttestationError(f"cannot read {path}: {error}") from error
    if not isinstance(value, list):
        raise AttestationError(f"{path} is not a JSON array")
    return value


def _campaign_rows(index_path: Path) -> list[tuple[str, str, Path]]:
    try:
        lines = index_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise AttestationError(f"cannot read {index_path}: {error}") from error
    rows: list[tuple[str, str, Path]] = []
    for line_number, line in enumerate(lines, start=1):
        fields = line.split("\t")
        if len(fields) != 3 or not all(fields):
            raise AttestationError(
                f"{index_path}:{line_number} must contain label, arm, and run directory"
            )
        label, arm, run_directory = fields
        if arm not in {"online-only", "stock", "planner-native"}:
            raise AttestationError(
                f"{index_path}:{line_number} has unknown arm {arm!r}"
            )
        rows.append((label, arm, Path(run_directory).expanduser().resolve()))
    if not rows:
        raise AttestationError(f"{index_path} contains no campaign runs")
    return rows


def _evidence_root(run_directory: Path, indexed_arm: str) -> Path:
    metadata_path = run_directory / "metadata.json"
    metadata = _read_object(metadata_path)
    arm = metadata.get("arm")
    if arm != indexed_arm:
        raise AttestationError(
            f"{metadata_path} arm {arm!r} does not match runs.tsv arm {indexed_arm!r}"
        )
    if arm == "online-only":
        return run_directory / "online-evidence"

    child_root = run_directory / "batch-harness" / "results" / "raw"
    children = (
        sorted(path for path in child_root.iterdir() if path.is_dir())
        if child_root.is_dir()
        else []
    )
    if len(children) != 1:
        raise AttestationError(
            f"{run_directory} expected exactly one nested Batch run, found {len(children)}"
        )
    return children[0]


def resolve_attestation(
    index_path: Path,
    dynamo_image: str,
    resolved_image_id: str | None,
) -> str | None:
    """Return IMAGE_REF=IMAGE_ID only when DYNAMO_IMAGE was sanitized."""
    if not dynamo_image or any(character.isspace() for character in dynamo_image):
        raise AttestationError("DYNAMO_IMAGE must be a non-empty image reference")

    redacted_refs: set[str] = set()
    for _label, arm, run_directory in _campaign_rows(index_path):
        evidence_root = _evidence_root(run_directory, arm)
        for phase in ("start", "end"):
            inventory_path = evidence_root / "kubernetes" / phase / "images.json"
            for item_number, item in enumerate(_read_array(inventory_path), start=1):
                if not isinstance(item, dict):
                    raise AttestationError(
                        f"{inventory_path} item {item_number} is not an object"
                    )
                image_id = item.get("image_id")
                if not (isinstance(image_id, str) and image_id.startswith("<redacted")):
                    continue
                image_ref = item.get("image")
                if not isinstance(image_ref, str) or not image_ref:
                    raise AttestationError(
                        f"{inventory_path} item {item_number} has a sanitized imageID "
                        "without an image reference"
                    )
                redacted_refs.add(image_ref)

    unknown_refs = sorted(redacted_refs - {dynamo_image})
    if unknown_refs:
        raise AttestationError(
            "campaign contains sanitized imageIDs for unsupported image refs: "
            + ", ".join(unknown_refs)
        )
    if dynamo_image not in redacted_refs:
        return None

    if not resolved_image_id:
        raise AttestationError(
            "the exact DYNAMO_IMAGE has sanitized imageIDs; set "
            "DYNAMO_IMAGE_ID_ATTESTATION to its resolved imageID only"
        )
    if (
        resolved_image_id.startswith("<redacted")
        or "=" in resolved_image_id
        or any(character.isspace() for character in resolved_image_id)
    ):
        raise AttestationError(
            "DYNAMO_IMAGE_ID_ATTESTATION must contain one resolved imageID only, "
            "not IMAGE_REF=IMAGE_ID or a redacted value"
        )
    return f"{dynamo_image}={resolved_image_id}"


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-index", required=True, type=Path)
    parser.add_argument("--dynamo-image", required=True)
    parser.add_argument("--resolved-image-id")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Print the compiler attestation, or print nothing when none is needed."""
    args = build_parser().parse_args(argv)
    try:
        attestation = resolve_attestation(
            args.runs_index,
            args.dynamo_image,
            args.resolved_image_id,
        )
    except AttestationError as error:
        raise SystemExit(
            f"cannot resolve campaign imageID attestation: {error}"
        ) from error
    if attestation is not None:
        print(attestation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
