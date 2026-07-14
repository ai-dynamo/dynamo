# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audit Qwen2.5-VL image workload JSONLs, JPEGs, hashes, and token lengths."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

from generate_workload import CUSTOM_TEMPLATE_TOKEN_DELTA, _calculate_native_isl
from PIL import Image
from transformers import AutoProcessor


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def validate_workload(root: Path) -> dict[str, Any]:
    manifest_path = root / "workload_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rates = [int(rate) for rate in manifest["rates"]]
    requests = int(manifest["requests_per_rate"])
    target_isl = int(manifest["target_isl"])
    encoding = manifest["encoding"]

    encoded_hashes: set[str] = set()
    decoded_hashes: set[str] = set()
    image_paths_by_rate: dict[int, list[str]] = {rate: [] for rate in rates}
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
        image_paths_by_rate[int(record["rate"])].append(str(path))
        sizes.append(len(raw))

    expected_images = len(rates) * requests
    if len(encoded_hashes) != expected_images or len(decoded_hashes) != expected_images:
        raise AssertionError("global image uniqueness check failed")

    processor = AutoProcessor.from_pretrained(manifest["model"])
    seen_paths: set[str] = set()
    for rate in rates:
        native_path = root / f"image_native_qps{rate}_{requests}_isl{target_isl}.jsonl"
        custom_path = root / f"image_custom_qps{rate}_{requests}_isl{target_isl}.jsonl"
        native = _read_jsonl(native_path)
        custom = _read_jsonl(custom_path)
        if len(native) != requests or len(custom) != requests:
            raise AssertionError(f"wrong request count for QPS {rate}")
        for dataset_path, rows in ((native_path, native), (custom_path, custom)):
            expected_file = manifest["files"][dataset_path.name]
            if _sha256(dataset_path) != expected_file["sha256"]:
                raise AssertionError(f"JSONL hash mismatch: {dataset_path}")
            if len({row["session_id"] for row in rows}) != requests:
                raise AssertionError(f"session IDs are not unique: {dataset_path}")
            if len({row["text"] for row in rows}) != 1:
                raise AssertionError(f"prompt is not identical: {dataset_path}")
        native_images = [row["image"] for row in native]
        custom_images = [row["image"] for row in custom]
        if native_images != custom_images:
            raise AssertionError(f"runtime image ordering differs at QPS {rate}")
        if native_images != image_paths_by_rate[rate]:
            raise AssertionError(f"manifest image ordering differs at QPS {rate}")
        if seen_paths.intersection(native_images):
            raise AssertionError(f"image pools overlap at QPS {rate}")
        seen_paths.update(native_images)

        with Image.open(native_images[0]) as encoded:
            image = encoded.convert("RGB")
        native_isl = _calculate_native_isl(processor, native[0]["text"], image)
        custom_isl = (
            _calculate_native_isl(processor, custom[0]["text"], image)
            - CUSTOM_TEMPLATE_TOKEN_DELTA
        )
        if native_isl != target_isl or custom_isl != target_isl:
            raise AssertionError(
                f"ISL calibration failed at QPS {rate}: "
                f"native={native_isl}, custom={custom_isl}, expected={target_isl}"
            )

    if manifest["unique_encoded_sha256"] != expected_images:
        raise AssertionError("manifest encoded uniqueness count is wrong")
    if manifest["unique_decoded_rgb_sha256"] != expected_images:
        raise AssertionError("manifest decoded uniqueness count is wrong")
    expected_sizes = manifest["file_size_bytes"]
    observed_sizes = {
        "min": min(sizes),
        "mean": statistics.mean(sizes),
        "median": statistics.median(sizes),
        "max": max(sizes),
    }
    if observed_sizes != expected_sizes:
        raise AssertionError("manifest file-size statistics are wrong")

    return {
        "manifest_sha256": _sha256(manifest_path),
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
