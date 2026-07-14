# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate the Qwen3-VL custom-encoder QPS benchmark workload."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from PIL import Image
from transformers import AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.multimodal.jsonl.generate_images import (  # noqa: E402
    JPEG_TARGET_MAX_BYTES,
    JPEG_TARGET_MIN_BYTES,
    generate_target_sized_jpeg_pool,
)

MODEL = "Qwen/Qwen3-VL-2B-Instruct"
RATES = (16, 24, 32)
REQUESTS_PER_RATE = 1000
TARGET_ISL = 515
BASE_PROMPT = "Classify the image and briefly explain the label."
CUSTOM_TEMPLATE_TOKEN_DELTA = 2


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> dict[str, Any]:
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, separators=(",", ":")) + "\n")
    return {"path": str(path.resolve()), "rows": len(rows), "sha256": _sha256(path)}


def _calculate_native_isl(processor: Any, prompt: str, image: Image.Image) -> int:
    rendered = processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(text=[rendered], images=[image], return_tensors="pt")
    return len(inputs.input_ids[0])


def _calibrate_prompt(
    processor: Any,
    image: Image.Image,
    target_isl: int,
    template_token_delta: int,
) -> tuple[str, int]:
    """Find an identical filler prompt whose rendered ISL is exact."""
    base_isl = (
        _calculate_native_isl(processor, BASE_PROMPT, image) - template_token_delta
    )
    one_repeat_isl = (
        _calculate_native_isl(processor, BASE_PROMPT + " benchmark", image)
        - template_token_delta
    )
    step = one_repeat_isl - base_isl
    if step <= 0:
        raise RuntimeError("benchmark filler did not increase token count")
    estimated_repeats = max(0, (target_isl - base_isl) // step)
    for repeat_count in range(max(0, estimated_repeats - 2), estimated_repeats + 4):
        prompt = BASE_PROMPT + " benchmark" * repeat_count
        isl = _calculate_native_isl(processor, prompt, image) - template_token_delta
        if isl == target_isl:
            return prompt, isl
    raise RuntimeError(
        f"could not calibrate target ISL {target_isl} with token delta "
        f"{template_token_delta}"
    )


def generate_workload(
    output_dir: Path,
    model: str = MODEL,
    rates: tuple[int, ...] = RATES,
    requests_per_rate: int = REQUESTS_PER_RATE,
    target_isl: int = TARGET_ISL,
    seed: int = 42,
    min_image_bytes: int = JPEG_TARGET_MIN_BYTES,
    max_image_bytes: int = JPEG_TARGET_MAX_BYTES,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_root = output_dir / "images"
    processor = AutoProcessor.from_pretrained(model)

    files: dict[str, dict[str, Any]] = {}
    image_records: list[dict[str, Any]] = []
    all_encoded_hashes: set[str] = set()
    all_decoded_hashes: set[str] = set()
    native_prompt = ""
    custom_prompt = ""

    for rate_index, rate in enumerate(rates):
        records = generate_target_sized_jpeg_pool(
            pool_size=requests_per_rate,
            image_dir=image_root / f"qps{rate}",
            seed=seed,
            min_bytes=min_image_bytes,
            max_bytes=max_image_bytes,
            start_index=rate_index * requests_per_rate,
        )
        for record in records:
            record["rate"] = rate
            encoded_hash = str(record["encoded_sha256"])
            decoded_hash = str(record["decoded_rgb_sha256"])
            if encoded_hash in all_encoded_hashes or decoded_hash in all_decoded_hashes:
                raise RuntimeError(
                    f"image pool is not globally unique: {record['path']}"
                )
            all_encoded_hashes.add(encoded_hash)
            all_decoded_hashes.add(decoded_hash)
            image_records.append(record)

        if not native_prompt:
            with Image.open(records[0]["path"]) as encoded:
                calibration_image = encoded.convert("RGB")
            native_prompt, native_isl = _calibrate_prompt(
                processor, calibration_image, target_isl, template_token_delta=0
            )
            custom_prompt, custom_isl = _calibrate_prompt(
                processor,
                calibration_image,
                target_isl,
                template_token_delta=CUSTOM_TEMPLATE_TOKEN_DELTA,
            )
            if native_isl != target_isl or custom_isl != target_isl:
                raise AssertionError(
                    "prompt calibration did not reach exact target ISL"
                )

        native_rows: list[dict[str, str]] = []
        custom_rows: list[dict[str, str]] = []
        for request_index, record in enumerate(records):
            session_id = f"qps{rate}-request-{request_index:04d}"
            common = {"session_id": session_id, "image": str(record["path"])}
            native_rows.append({**common, "text": native_prompt})
            custom_rows.append({**common, "text": custom_prompt})

        native_name = (
            f"image_native_qps{rate}_{requests_per_rate}_isl{target_isl}.jsonl"
        )
        custom_name = (
            f"image_custom_qps{rate}_{requests_per_rate}_isl{target_isl}.jsonl"
        )
        files[native_name] = _write_jsonl(output_dir / native_name, native_rows)
        files[custom_name] = _write_jsonl(output_dir / custom_name, custom_rows)

    sizes = [int(record["size_bytes"]) for record in image_records]
    expected_images = len(rates) * requests_per_rate
    if (
        len(all_encoded_hashes) != expected_images
        or len(all_decoded_hashes) != expected_images
    ):
        raise RuntimeError("generated image pools are not globally unique")

    manifest = {
        "model": model,
        "rates": list(rates),
        "requests_per_rate": requests_per_rate,
        "seed": seed,
        "target_isl": target_isl,
        "custom_template_token_delta": CUSTOM_TEMPLATE_TOKEN_DELTA,
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
        "unique_encoded_sha256": len(all_encoded_hashes),
        "unique_decoded_rgb_sha256": len(all_decoded_hashes),
        "prompts": {"native": native_prompt, "custom": custom_prompt},
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
                "file_size_bytes": manifest["file_size_bytes"],
                "target_isl": target_isl,
            },
            indent=2,
        )
    )
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / ".data",
    )
    parser.add_argument("--rates", type=int, nargs="+", default=list(RATES))
    parser.add_argument("--requests-per-rate", type=int, default=REQUESTS_PER_RATE)
    parser.add_argument("--target-isl", type=int, default=TARGET_ISL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-image-kib", type=int, default=50)
    parser.add_argument("--max-image-kib", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_workload(
        output_dir=args.output_dir,
        model=args.model,
        rates=tuple(args.rates),
        requests_per_rate=args.requests_per_rate,
        target_isl=args.target_isl,
        seed=args.seed,
        min_image_bytes=args.min_image_kib * 1024,
        max_image_bytes=args.max_image_kib * 1024,
    )


if __name__ == "__main__":
    main()
