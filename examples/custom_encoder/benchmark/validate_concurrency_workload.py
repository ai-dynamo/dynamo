# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audit the disjoint 8/16/32 Qwen2.5 custom-encoder image pools."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.custom_encoder.benchmark.generate_workload import (  # noqa: E402
    _calculate_custom_isl_components,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def validate_workload(root: Path) -> dict[str, Any]:
    manifest_path = root / "workload_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("axis") != "concurrency":
        raise AssertionError("workload axis must be concurrency")
    concurrencies = [int(value) for value in manifest["concurrencies"]]
    requests = int(manifest["requests_per_concurrency"])
    target_isl = int(manifest["target_isl"])
    encoding = manifest["encoding"]

    encoded_hashes: set[str] = set()
    decoded_hashes: set[str] = set()
    image_paths: dict[int, list[str]] = {value: [] for value in concurrencies}
    sizes: list[int] = []
    for record in manifest["images"]:
        path = Path(record["path"])
        raw = path.read_bytes()
        if not encoding["min_bytes"] <= len(raw) <= encoding["max_bytes"]:
            raise AssertionError(f"JPEG size out of range: {path} ({len(raw)} bytes)")
        if len(raw) != record["size_bytes"]:
            raise AssertionError(f"size mismatch: {path}")
        encoded_hash = hashlib.sha256(raw).hexdigest()
        if encoded_hash != record["encoded_sha256"]:
            raise AssertionError(f"encoded hash mismatch: {path}")
        with Image.open(path) as encoded:
            if encoded.format != "JPEG":
                raise AssertionError(f"not a JPEG: {path}")
            decoded = encoded.convert("RGB")
            if decoded.size != (500, 500):
                raise AssertionError(f"wrong decoded dimensions: {path}")
            decoded_hash = hashlib.sha256(decoded.tobytes()).hexdigest()
        if decoded_hash != record["decoded_rgb_sha256"]:
            raise AssertionError(f"decoded hash mismatch: {path}")
        if encoded_hash in encoded_hashes or decoded_hash in decoded_hashes:
            raise AssertionError(f"duplicate image: {path}")
        encoded_hashes.add(encoded_hash)
        decoded_hashes.add(decoded_hash)
        image_paths[int(record["concurrency"])].append(str(path))
        sizes.append(len(raw))

    expected_images = len(concurrencies) * requests
    if len(encoded_hashes) != expected_images or len(decoded_hashes) != expected_images:
        raise AssertionError("global image uniqueness check failed")

    encoder_processor = AutoProcessor.from_pretrained(manifest["encoder_model"])
    decoder_tokenizer = AutoTokenizer.from_pretrained(manifest["decoder_model"])
    seen_paths: set[str] = set()
    for concurrency in concurrencies:
        dataset_path = root / (
            f"image_custom_concurrency{concurrency}_{requests}_isl{target_isl}.jsonl"
        )
        rows = _read_jsonl(dataset_path)
        if len(rows) != requests:
            raise AssertionError(f"wrong request count at concurrency {concurrency}")
        expected_file = manifest["files"][dataset_path.name]
        if _sha256(dataset_path) != expected_file["sha256"]:
            raise AssertionError(f"JSONL hash mismatch: {dataset_path}")
        if len({row["session_id"] for row in rows}) != requests:
            raise AssertionError(f"session IDs are not unique: {dataset_path}")
        if len({row["text"] for row in rows}) != 1:
            raise AssertionError(f"prompt is not identical: {dataset_path}")
        paths = [row["image"] for row in rows]
        if paths != image_paths[concurrency]:
            raise AssertionError(
                f"manifest image ordering differs at concurrency {concurrency}"
            )
        if seen_paths.intersection(paths):
            raise AssertionError(f"image pools overlap at concurrency {concurrency}")
        seen_paths.update(paths)

        with Image.open(paths[0]) as encoded:
            image = encoded.convert("RGB")
        isl = _calculate_custom_isl_components(
            decoder_tokenizer,
            encoder_processor.image_processor,
            rows[0]["text"],
            image,
        )
        if isl != target_isl:
            raise AssertionError(
                f"ISL calibration failed at concurrency {concurrency}: "
                f"observed={isl}, expected={target_isl}"
            )

    if manifest["unique_encoded_sha256"] != expected_images:
        raise AssertionError("manifest encoded uniqueness count is wrong")
    if manifest["unique_decoded_rgb_sha256"] != expected_images:
        raise AssertionError("manifest decoded uniqueness count is wrong")
    observed_sizes = {
        "min": min(sizes),
        "mean": statistics.mean(sizes),
        "median": statistics.median(sizes),
        "max": max(sizes),
    }
    if observed_sizes != manifest["file_size_bytes"]:
        raise AssertionError("manifest file-size statistics are wrong")

    return {
        "manifest_sha256": _sha256(manifest_path),
        "axis": "concurrency",
        "values": concurrencies,
        "images": expected_images,
        "encoded_unique": len(encoded_hashes),
        "decoded_unique": len(decoded_hashes),
        "target_isl": target_isl,
        "file_size_bytes": observed_sizes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workload_dir", type=Path)
    args = parser.parse_args()
    result = validate_workload(args.workload_dir.resolve())
    print("WORKLOAD_AUDIT=PASS")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
