# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate the performance-only Qwen2.5 custom-encoder concurrency workload."""

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

from benchmarks.multimodal.jsonl.generate_images import (  # noqa: E402
    JPEG_TARGET_MAX_BYTES,
    JPEG_TARGET_MIN_BYTES,
    generate_target_sized_jpeg_pool,
)
from examples.custom_encoder.benchmark.generate_workload import (  # noqa: E402
    TARGET_ISL,
    _calculate_custom_isl_components,
    _calibrate_prompt,
)

DECODER_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ENCODER_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
CONCURRENCIES = (8, 16, 32)
REQUESTS_PER_CONCURRENCY = 1000


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> dict[str, Any]:
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, separators=(",", ":")) + "\n")
    return {"path": str(path.resolve()), "rows": len(rows), "sha256": _sha256(path)}


def generate_workload(
    output_dir: Path,
    decoder_model: str = DECODER_MODEL,
    encoder_model: str = ENCODER_MODEL,
    concurrencies: tuple[int, ...] = CONCURRENCIES,
    requests_per_concurrency: int = REQUESTS_PER_CONCURRENCY,
    target_isl: int = TARGET_ISL,
    seed: int = 42,
    min_image_bytes: int = JPEG_TARGET_MIN_BYTES,
    max_image_bytes: int = JPEG_TARGET_MAX_BYTES,
) -> Path:
    if not concurrencies or any(value < 1 for value in concurrencies):
        raise ValueError("concurrencies must contain positive integers")
    output_dir.mkdir(parents=True, exist_ok=True)
    image_root = output_dir / "images"
    encoder_processor = AutoProcessor.from_pretrained(encoder_model)
    decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_model)

    def calculate_custom_isl(_processor: Any, prompt: str, image: Image.Image) -> int:
        return _calculate_custom_isl_components(
            decoder_tokenizer,
            encoder_processor.image_processor,
            prompt,
            image,
        )

    files: dict[str, dict[str, Any]] = {}
    image_records: list[dict[str, Any]] = []
    encoded_hashes: set[str] = set()
    decoded_hashes: set[str] = set()
    calibrated_prompt = ""

    for value_index, concurrency in enumerate(concurrencies):
        records = generate_target_sized_jpeg_pool(
            pool_size=requests_per_concurrency,
            image_dir=image_root / f"concurrency{concurrency}",
            seed=seed,
            min_bytes=min_image_bytes,
            max_bytes=max_image_bytes,
            start_index=value_index * requests_per_concurrency,
        )
        for record in records:
            record["concurrency"] = concurrency
            encoded_hash = str(record["encoded_sha256"])
            decoded_hash = str(record["decoded_rgb_sha256"])
            if encoded_hash in encoded_hashes or decoded_hash in decoded_hashes:
                raise RuntimeError(
                    f"image pool is not globally unique: {record['path']}"
                )
            encoded_hashes.add(encoded_hash)
            decoded_hashes.add(decoded_hash)
            image_records.append(record)

        if not calibrated_prompt:
            with Image.open(records[0]["path"]) as encoded:
                calibration_image = encoded.convert("RGB")
            calibrated_prompt, observed_isl = _calibrate_prompt(
                encoder_processor,
                calibration_image,
                target_isl,
                calculate_isl=calculate_custom_isl,
            )
            if observed_isl != target_isl:
                raise AssertionError(
                    "prompt calibration did not reach exact target ISL"
                )

        rows = [
            {
                "session_id": f"concurrency{concurrency}-request-{index:04d}",
                "image": str(record["path"]),
                "text": calibrated_prompt,
            }
            for index, record in enumerate(records)
        ]
        name = (
            f"image_custom_concurrency{concurrency}_"
            f"{requests_per_concurrency}_isl{target_isl}.jsonl"
        )
        files[name] = _write_jsonl(output_dir / name, rows)

    expected_images = len(concurrencies) * requests_per_concurrency
    if len(encoded_hashes) != expected_images or len(decoded_hashes) != expected_images:
        raise RuntimeError("generated image pools are not globally unique")
    sizes = [int(record["size_bytes"]) for record in image_records]
    manifest = {
        "axis": "concurrency",
        "concurrencies": list(concurrencies),
        "decoder_model": decoder_model,
        "encoder_model": encoder_model,
        "requests_per_concurrency": requests_per_concurrency,
        "seed": seed,
        "target_isl": target_isl,
        "performance_only_adapter": {
            "native_vision_width": 2048,
            "decoder_hidden_width": 1536,
            "operation": "truncate the fully computed vision output to its first 1536 columns",
            "quality_or_parity_claim": False,
        },
        "custom_isl_calibration": (
            "Qwen2.5-1.5B decoder text tokens minus one image placeholder plus "
            "Qwen2.5-VL-3B processor-derived merged image tokens"
        ),
        "decoded_image": {"mode": "RGB", "width": 500, "height": 500},
        "encoding": {
            "format": "JPEG",
            "quality": 85,
            "subsampling": "4:2:0",
            "min_bytes": min_image_bytes,
            "max_bytes": max_image_bytes,
        },
        "file_size_bytes": {
            "min": min(sizes),
            "mean": statistics.mean(sizes),
            "median": statistics.median(sizes),
            "max": max(sizes),
        },
        "unique_encoded_sha256": len(encoded_hashes),
        "unique_decoded_rgb_sha256": len(decoded_hashes),
        "prompt": calibrated_prompt,
        "files": files,
        "images": image_records,
    }
    manifest_path = output_dir / "workload_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest": str(manifest_path.resolve()),
                "images": expected_images,
                "concurrencies": list(concurrencies),
                "target_isl": target_isl,
            },
            indent=2,
        )
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoder-model", default=DECODER_MODEL)
    parser.add_argument("--encoder-model", default=ENCODER_MODEL)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).parent / ".concurrency-data"
    )
    parser.add_argument(
        "--concurrencies", type=int, nargs="+", default=list(CONCURRENCIES)
    )
    parser.add_argument(
        "--requests-per-concurrency", type=int, default=REQUESTS_PER_CONCURRENCY
    )
    parser.add_argument("--target-isl", type=int, default=TARGET_ISL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-image-kib", type=int, default=50)
    parser.add_argument("--max-image-kib", type=int, default=60)
    args = parser.parse_args()
    generate_workload(
        output_dir=args.output_dir.resolve(),
        decoder_model=args.decoder_model,
        encoder_model=args.encoder_model,
        concurrencies=tuple(args.concurrencies),
        requests_per_concurrency=args.requests_per_concurrency,
        target_isl=args.target_isl,
        seed=args.seed,
        min_image_bytes=args.min_image_kib * 1024,
        max_image_bytes=args.max_image_kib * 1024,
    )


if __name__ == "__main__":
    main()
