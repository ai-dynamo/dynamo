# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate and audit the fixed-shape Qwen custom-encoder proxy workload."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DECODER_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ENCODER_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
CONCURRENCIES = tuple(range(1, 11))
REQUESTS = 100
UNIQUE_IMAGES = 9
TARGET_ISL = 644
TARGET_OSL = 7
SEED = 42
JPEG_MIN_BYTES = 50 * 1024
JPEG_MAX_BYTES = 60 * 1024
BASE_PROMPT = "Classify the image and briefly explain the label."
CUSTOM_IMAGE_TOKEN = "<|image_pad|>"
CUSTOM_CHAT_TEMPLATE = (
    REPO_ROOT / "examples/custom_encoder/templates/qwen_vl.jinja"
).read_text(encoding="utf-8")
INPUT_NAME = f"image_custom_{REQUESTS}_isl{TARGET_ISL}.jsonl"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _encode_resampled_noise_jpeg(
    noise: Image.Image,
    texture_side: int,
    image_size: tuple[int, int],
    quality: int,
) -> bytes:
    image = noise.resize(
        (texture_side, texture_side), Image.Resampling.BILINEAR
    ).resize(image_size, Image.Resampling.BICUBIC)
    encoded = io.BytesIO()
    image.save(
        encoded,
        format="JPEG",
        quality=quality,
        optimize=True,
        subsampling=2,
    )
    return encoded.getvalue()


def _generate_jpeg(
    path: Path,
    seed: int,
    min_bytes: int = JPEG_MIN_BYTES,
    max_bytes: int = JPEG_MAX_BYTES,
) -> dict[str, Any]:
    image_size = (500, 500)
    pixels = np.random.default_rng(seed).integers(
        0, 256, (image_size[1], image_size[0], 3), dtype=np.uint8
    )
    noise = Image.fromarray(pixels)
    target_bytes = (min_bytes + max_bytes) // 2
    candidates: list[tuple[int, bytes]] = []

    def encode(texture_side: int) -> bytes:
        payload = _encode_resampled_noise_jpeg(
            noise, texture_side, image_size, quality=85
        )
        candidates.append((texture_side, payload))
        return payload

    payload = encode(180)
    if not min_bytes <= len(payload) <= max_bytes:
        lower, upper = 8, min(image_size)
        while lower <= upper:
            texture_side = (lower + upper) // 2
            payload = encode(texture_side)
            if min_bytes <= len(payload) <= max_bytes:
                break
            if len(payload) < min_bytes:
                lower = texture_side + 1
            else:
                upper = texture_side - 1

    texture_side, payload = min(
        candidates, key=lambda candidate: abs(len(candidate[1]) - target_bytes)
    )
    if not min_bytes <= len(payload) <= max_bytes:
        raise RuntimeError(
            f"could not generate JPEG in [{min_bytes}, {max_bytes}] bytes; "
            f"closest was {len(payload)} bytes"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    with Image.open(io.BytesIO(payload)) as encoded:
        decoded = encoded.convert("RGB")
        decoded_hash = hashlib.sha256(decoded.tobytes()).hexdigest()
    return {
        "path": str(path.resolve()),
        "width": 500,
        "height": 500,
        "size_bytes": len(payload),
        "jpeg_quality": 85,
        "texture_side": texture_side,
        "encoded_sha256": hashlib.sha256(payload).hexdigest(),
        "decoded_rgb_sha256": decoded_hash,
    }


def _calculate_custom_isl_components(
    tokenizer: Any,
    image_processor: Any,
    prompt: str,
    image: Image.Image,
) -> int:
    rendered = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        chat_template=CUSTOM_CHAT_TEMPLATE,
        tokenize=False,
        add_generation_prompt=True,
    )
    text_ids = tokenizer(rendered, add_special_tokens=False).input_ids
    image_token_id = tokenizer.convert_tokens_to_ids(CUSTOM_IMAGE_TOKEN)
    if text_ids.count(image_token_id) != 1:
        raise RuntimeError("custom template must emit exactly one image token")
    image_inputs = image_processor(images=[image], return_tensors="pt")
    grid = image_inputs["image_grid_thw"][0]
    merge_size = int(image_processor.merge_size)
    image_tokens = int(grid.prod().item()) // merge_size**2
    return len(text_ids) - 1 + image_tokens


def _calibrate_prompt(
    target_isl: int,
    calculate_isl: Callable[[str], int],
) -> tuple[str, int]:
    base_isl = calculate_isl(BASE_PROMPT)
    one_repeat_isl = calculate_isl(BASE_PROMPT + " benchmark")
    step = one_repeat_isl - base_isl
    if step <= 0:
        raise RuntimeError("benchmark filler did not increase token count")
    estimated_repeats = max(0, (target_isl - base_isl) // step)
    for repeats in range(max(0, estimated_repeats - 4), estimated_repeats + 8):
        prompt = BASE_PROMPT + " benchmark" * repeats
        observed = calculate_isl(prompt)
        if observed == target_isl:
            return prompt, observed
    raise RuntimeError(f"could not calibrate exact target ISL {target_isl}")


def _request_schedule(image_paths: list[str], requests: int, seed: int) -> list[str]:
    if not image_paths:
        raise ValueError("image_paths must not be empty")
    if requests < len(image_paths):
        raise ValueError("requests must cover every unique image")
    schedule = [image_paths[index % len(image_paths)] for index in range(requests)]
    random.Random(seed).shuffle(schedule)
    return schedule


def generate_workload(
    output_dir: Path,
    decoder_model: str = DECODER_MODEL,
    encoder_model: str = ENCODER_MODEL,
    requests: int = REQUESTS,
    unique_images: int = UNIQUE_IMAGES,
    target_isl: int = TARGET_ISL,
    seed: int = SEED,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(decoder_model)
    processor = AutoProcessor.from_pretrained(encoder_model)

    records = [
        _generate_jpeg(
            output_dir / "images" / f"image_{index:02d}_500x500.jpg",
            seed + index,
        )
        for index in range(unique_images)
    ]
    encoded_hashes = {str(record["encoded_sha256"]) for record in records}
    decoded_hashes = {str(record["decoded_rgb_sha256"]) for record in records}
    if len(encoded_hashes) != unique_images or len(decoded_hashes) != unique_images:
        raise RuntimeError("generated image pool is not globally unique")

    with Image.open(str(records[0]["path"])) as encoded:
        calibration_image = encoded.convert("RGB")

    def calculate_isl(prompt: str) -> int:
        return _calculate_custom_isl_components(
            tokenizer, processor.image_processor, prompt, calibration_image
        )

    prompt, observed_isl = _calibrate_prompt(target_isl, calculate_isl)
    for record in records:
        with Image.open(str(record["path"])) as encoded:
            image = encoded.convert("RGB")
        if (
            _calculate_custom_isl_components(
                tokenizer, processor.image_processor, prompt, image
            )
            != target_isl
        ):
            raise RuntimeError("fixed-shape image produced a different ISL")

    schedule = _request_schedule(
        [str(record["path"]) for record in records], requests, seed
    )
    rows = [
        {
            "session_id": f"request-{index:04d}",
            "image": image_path,
            "text": prompt,
        }
        for index, image_path in enumerate(schedule)
    ]
    input_path = output_dir / f"image_custom_{requests}_isl{target_isl}.jsonl"
    with input_path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, separators=(",", ":")) + "\n")

    sizes = [int(record["size_bytes"]) for record in records]
    occurrence_counts = Counter(schedule)
    manifest = {
        "axis": "concurrency",
        "concurrencies": list(CONCURRENCIES),
        "decoder_model": decoder_model,
        "encoder_model": encoder_model,
        "requests_per_concurrency": requests,
        "warmup_requests": 20,
        "unique_images": unique_images,
        "seed": seed,
        "target_isl": target_isl,
        "target_osl": TARGET_OSL,
        "prompt": prompt,
        "prompt_policy": "byte-identical calibrated synthetic prompt",
        "decoded_image": {"mode": "RGB", "width": 500, "height": 500},
        "encoding": {
            "format": "JPEG",
            "quality": 85,
            "subsampling": "4:2:0",
            "min_bytes": JPEG_MIN_BYTES,
            "max_bytes": JPEG_MAX_BYTES,
        },
        "file_size_bytes": {
            "min": min(sizes),
            "mean": statistics.mean(sizes),
            "median": statistics.median(sizes),
            "max": max(sizes),
        },
        "unique_encoded_sha256": len(encoded_hashes),
        "unique_decoded_rgb_sha256": len(decoded_hashes),
        "occurrences": dict(sorted(occurrence_counts.items())),
        "input": {
            "path": str(input_path.resolve()),
            "rows": len(rows),
            "sha256": _sha256(input_path),
        },
        "images": records,
        "observed_calibration_isl": observed_isl,
    }
    manifest_path = output_dir / "workload_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"workload={input_path.resolve()} images={unique_images} "
        f"requests={requests} isl={observed_isl}"
    )
    return manifest_path


def validate_workload(root: Path) -> dict[str, Any]:
    manifest_path = root / "workload_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    requests = int(manifest["requests_per_concurrency"])
    unique_images = int(manifest["unique_images"])
    target_isl = int(manifest["target_isl"])
    input_path = Path(manifest["input"]["path"])
    rows = [
        json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines()
    ]
    if len(rows) != requests or manifest["input"]["sha256"] != _sha256(input_path):
        raise AssertionError("input JSONL count or hash mismatch")
    if len({row["session_id"] for row in rows}) != requests:
        raise AssertionError("session IDs must be unique")
    if len({row["text"] for row in rows}) != 1:
        raise AssertionError("prompt must be byte-identical across requests")

    image_paths = {str(record["path"]) for record in manifest["images"]}
    if len(image_paths) != unique_images:
        raise AssertionError("manifest image count is wrong")
    occurrence_counts = Counter(str(row["image"]) for row in rows)
    if set(occurrence_counts) != image_paths:
        raise AssertionError("JSONL does not use exactly the manifest image pool")
    expected_counts = sorted(
        requests // unique_images + (1 if index < requests % unique_images else 0)
        for index in range(unique_images)
    )
    actual_counts = sorted(occurrence_counts.values())
    if actual_counts != expected_counts:
        raise AssertionError(f"unexpected image reuse distribution: {actual_counts}")

    encoded_hashes: set[str] = set()
    decoded_hashes: set[str] = set()
    tokenizer = AutoTokenizer.from_pretrained(manifest["decoder_model"])
    processor = AutoProcessor.from_pretrained(manifest["encoder_model"])
    prompt = str(manifest["prompt"])
    for record in manifest["images"]:
        path = Path(record["path"])
        payload = path.read_bytes()
        if not JPEG_MIN_BYTES <= len(payload) <= JPEG_MAX_BYTES:
            raise AssertionError(f"JPEG size out of range: {path}")
        encoded_hash = hashlib.sha256(payload).hexdigest()
        if encoded_hash != record["encoded_sha256"]:
            raise AssertionError(f"encoded hash mismatch: {path}")
        with Image.open(path) as encoded:
            if encoded.format != "JPEG" or encoded.size != (500, 500):
                raise AssertionError(f"invalid JPEG shape or format: {path}")
            image = encoded.convert("RGB")
        decoded_hash = hashlib.sha256(image.tobytes()).hexdigest()
        if decoded_hash != record["decoded_rgb_sha256"]:
            raise AssertionError(f"decoded hash mismatch: {path}")
        if (
            _calculate_custom_isl_components(
                tokenizer, processor.image_processor, prompt, image
            )
            != target_isl
        ):
            raise AssertionError(f"ISL calibration mismatch: {path}")
        encoded_hashes.add(encoded_hash)
        decoded_hashes.add(decoded_hash)
    if len(encoded_hashes) != unique_images or len(decoded_hashes) != unique_images:
        raise AssertionError("image uniqueness audit failed")

    result = {
        "manifest_sha256": _sha256(manifest_path),
        "input_sha256": _sha256(input_path),
        "requests": requests,
        "images": unique_images,
        "target_isl": target_isl,
        "reuse_counts": actual_counts,
    }
    print("WORKLOAD_AUDIT=PASS")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--output-dir", type=Path, required=True)
    generate.add_argument("--decoder-model", default=DECODER_MODEL)
    generate.add_argument("--encoder-model", default=ENCODER_MODEL)
    validate = subparsers.add_parser("validate")
    validate.add_argument("workload_dir", type=Path)
    args = parser.parse_args()
    if args.command == "generate":
        generate_workload(
            args.output_dir.resolve(),
            decoder_model=args.decoder_model,
            encoder_model=args.encoder_model,
        )
    else:
        validate_workload(args.workload_dir.resolve())


if __name__ == "__main__":
    main()
