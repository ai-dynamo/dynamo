#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download images for benchmark."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
from pathlib import Path

DATASET_ID = "lmms-lab-encoder/DocVQA"
DATASET_REVISION = "539088ef8a8ada01ac8e2e6d4e372586748a265e"
SOURCE_SHARD = "DocVQA/validation-00000-of-00006.parquet"
SOURCE_SHARD_SHA256 = "a31507f0b700ac64f3ead52057c5dc3ccfb0baadcc62bc0d19d17159f080a4d8"
DEFAULT_IMAGE_COUNT = 50


class DatasetDownloadError(RuntimeError):
    """The source or output does not satisfy this small dataset contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_count(count: int) -> int:
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 50:
        raise DatasetDownloadError(f"image count must be between 1 and 50, got {count}")
    return count


def _verify_shard(path: Path, expected_sha256: str) -> None:
    if not path.is_file():
        raise DatasetDownloadError(f"parquet source is not a file: {path}")
    actual = _sha256_file(path)
    if actual != expected_sha256:
        raise DatasetDownloadError(
            f"parquet SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )


def download_pinned_shard(cache_dir: Path | None = None) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise DatasetDownloadError(
            "install huggingface_hub to download benchmark images"
        ) from exc

    path = Path(
        hf_hub_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            revision=DATASET_REVISION,
            filename=SOURCE_SHARD,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
    )
    _verify_shard(path, SOURCE_SHARD_SHA256)
    return path


def _reuse_existing(output_dir: Path, count: int, expected_sha256: str) -> bool:
    if not output_dir.exists():
        return False
    if not output_dir.is_dir():
        raise DatasetDownloadError(f"output path is not a directory: {output_dir}")
    if not any(output_dir.iterdir()):
        return False
    try:
        from PIL import Image

        manifest = [
            json.loads(line)
            for line in (output_dir / "manifest.jsonl").read_text().splitlines()
        ]
        if len(manifest) < count:
            raise ValueError(f"manifest contains only {len(manifest)} images")
        for index, row in enumerate(manifest[:count]):
            if row.get("dataset_id", row.get("source_dataset")) != DATASET_ID:
                raise ValueError("manifest dataset does not match pinned dataset")
            if row.get("dataset_revision") != DATASET_REVISION:
                raise ValueError("manifest revision does not match pinned dataset")
            if row.get("source_shard") != SOURCE_SHARD:
                raise ValueError("manifest shard does not match pinned dataset")
            shard_sha = row.get("source_shard_sha256")
            if "dataset_id" in row and shard_sha != expected_sha256:
                raise ValueError("manifest shard checksum does not match")
            path = output_dir / row["image_file"]
            expected_output_sha = row.get("output_sha256") or row.get(
                "request_image", {}
            ).get("sha256")
            if path != output_dir / "images" / f"{index:03d}.png":
                raise ValueError("manifest image order is not deterministic")
            if _sha256_file(path) != expected_output_sha:
                raise ValueError(f"image checksum mismatch: {path}")
            with Image.open(path) as image:
                image.load()
                if image.format != "PNG" or image.mode != "RGB":
                    raise ValueError("not a normalized PNG")
    except (ImportError, OSError, ValueError) as exc:
        raise DatasetDownloadError(
            f"output directory is non-empty but cannot be reused: {output_dir}: {exc}"
        ) from exc
    return True


def _extract(
    parquet_path: Path, output_dir: Path, count: int, source_shard_sha256: str
) -> list[dict[str, object]]:
    try:
        import pyarrow.parquet as pq
        from PIL import Image
    except ImportError as exc:
        raise DatasetDownloadError(
            "install pyarrow and Pillow to extract benchmark images"
        ) from exc

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    source_row = 0
    batches = pq.ParquetFile(parquet_path).iter_batches(columns=["image"])
    for batch in batches:
        for record in batch.to_pylist():
            image = record.get("image")
            raw = image.get("bytes") if isinstance(image, dict) else None
            if isinstance(raw, (bytearray, memoryview)):
                raw = bytes(raw)
            if not isinstance(raw, bytes) or not raw:
                raise DatasetDownloadError(
                    f"row {source_row} has no embedded image bytes"
                )
            source_sha = hashlib.sha256(raw).hexdigest()
            row = source_row
            source_row += 1
            if source_sha in seen:
                continue
            seen.add(source_sha)
            with Image.open(io.BytesIO(raw)) as source:
                source.load()
                source_mode = source.mode
                normalized = source.convert("RGB")
            path = images_dir / f"{len(rows):03d}.png"
            normalized.save(path, format="PNG", optimize=False, compress_level=9)
            rows.append(
                {
                    "index": len(rows),
                    "dataset_id": DATASET_ID,
                    "dataset_revision": DATASET_REVISION,
                    "source_shard": SOURCE_SHARD,
                    "source_shard_sha256": source_shard_sha256,
                    "source_row": row,
                    "source_sha256": source_sha,
                    "image_file": path.relative_to(output_dir).as_posix(),
                    "output_sha256": _sha256_file(path),
                    "width": normalized.width,
                    "height": normalized.height,
                    "source_mode": source_mode,
                }
            )
            if len(rows) == count:
                break
        if len(rows) == count:
            break
    if len(rows) != count:
        raise DatasetDownloadError(
            f"only found {len(rows)} unique images; requested {count}"
        )
    (output_dir / "manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return rows


def prepare_dataset(
    parquet_path: Path,
    output_dir: Path,
    *,
    count: int = DEFAULT_IMAGE_COUNT,
    expected_parquet_sha256: str = SOURCE_SHARD_SHA256,
) -> tuple[str, dict[str, object]]:
    count = validate_count(count)
    output_dir = Path(output_dir)
    if _reuse_existing(output_dir, count, expected_parquet_sha256):
        return "reused", {"count": count, "output_dir": str(output_dir)}
    _verify_shard(Path(parquet_path), expected_parquet_sha256)
    if output_dir.exists():
        output_dir.rmdir()
    output_dir.mkdir(parents=True)
    try:
        rows = _extract(Path(parquet_path), output_dir, count, expected_parquet_sha256)
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise
    return "created", {"count": count, "output_dir": str(output_dir), "images": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--count", default=DEFAULT_IMAGE_COUNT, type=int)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        count = validate_count(args.count)
        if _reuse_existing(args.output_dir, count, SOURCE_SHARD_SHA256):
            action = "reused"
        else:
            source = download_pinned_shard(args.cache_dir)
            action, _ = prepare_dataset(source, args.output_dir, count=count)
    except DatasetDownloadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"{action}: {count} PNG images in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
