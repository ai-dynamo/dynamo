# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate and audit the fixed-text mixed-image custom-encoder workload."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import statistics
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DECODER_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ENCODER_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
CONCURRENCY = 64
REQUESTS = 1000
TEXT_ISL = 644
TARGET_OSL = 7
SEED = 42
JPEG_MIN_BYTES = 50 * 1024
JPEG_MAX_BYTES = 60 * 1024
BENCHMARK_JPEG_TARGETS = {
    (300, 300): 7 * 1024,
    (500, 500): 35 * 1024,
}
BENCHMARK_JPEG_TOLERANCE = 512
DEFAULT_IMAGE_SIZE = 500
BENCHMARK_IMAGE_SIZE_COUNTS = ((300, 300, 500), (500, 500, 500))
BASE_PROMPT = "Classify the image and briefly explain the label."
CUSTOM_IMAGE_TOKEN = "<|image_pad|>"
CUSTOM_CHAT_TEMPLATE = (
    REPO_ROOT / "examples/custom_encoder/templates/qwen_vl.jinja"
).read_text(encoding="utf-8")
INPUT_NAME = f"image_custom_{REQUESTS}_textisl{TEXT_ISL}.jsonl"


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
    image_size: tuple[int, int] = (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE),
    min_bytes: int = JPEG_MIN_BYTES,
    max_bytes: int = JPEG_MAX_BYTES,
) -> dict[str, Any]:
    if min(image_size) < 1:
        raise ValueError("image dimensions must be positive")
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

    payload = encode(min(180, min(image_size)))
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
        "width": image_size[0],
        "height": image_size[1],
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


def _normalize_image_size_counts(
    image_size: int,
    unique_images: int,
    image_size_counts: tuple[tuple[int, int, int], ...] | None,
) -> tuple[tuple[int, int, int], ...]:
    if image_size_counts is None:
        image_size_counts = ((image_size, image_size, unique_images),)
    if not image_size_counts:
        raise ValueError("image_size_counts must not be empty")
    normalized: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int]] = set()
    for width, height, count in image_size_counts:
        if width < 1 or height < 1 or count < 1:
            raise ValueError("image dimensions and counts must be positive")
        dimensions = (width, height)
        if dimensions in seen:
            raise ValueError(f"duplicate image size: {width}x{height}")
        seen.add(dimensions)
        normalized.append((width, height, count))
    return tuple(normalized)


def _parse_image_size_count(value: str) -> tuple[int, int, int]:
    try:
        dimensions, count_text = value.rsplit(":", 1)
        width_text, height_text = dimensions.lower().split("x", 1)
        parsed = (int(width_text), int(height_text), int(count_text))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "expected WIDTHxHEIGHT:COUNT, for example 300x300:500"
        ) from exc
    if min(parsed) < 1:
        raise argparse.ArgumentTypeError("dimensions and count must be positive")
    return parsed


def generate_workload(
    output_dir: Path,
    decoder_model: str = DECODER_MODEL,
    encoder_model: str = ENCODER_MODEL,
    requests: int = REQUESTS,
    text_isl: int = TEXT_ISL,
    seed: int = SEED,
    image_size: int = DEFAULT_IMAGE_SIZE,
    image_size_counts: tuple[tuple[int, int, int], ...] | None = None,
) -> Path:
    if requests < 1:
        raise ValueError("requests must be positive")
    if text_isl < 1:
        raise ValueError("text_isl must be positive")
    normalized_sizes = _normalize_image_size_counts(
        image_size,
        requests,
        image_size_counts or BENCHMARK_IMAGE_SIZE_COUNTS,
    )
    unique_images = sum(count for _, _, count in normalized_sizes)
    if unique_images > requests:
        raise ValueError("unique images must not exceed requests")
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(decoder_model)
    processor = AutoProcessor.from_pretrained(encoder_model)
    records: list[dict[str, Any]] = []
    image_index = 0
    for width, height, count in normalized_sizes:
        target_bytes = BENCHMARK_JPEG_TARGETS.get((width, height))
        min_bytes = (
            target_bytes - BENCHMARK_JPEG_TOLERANCE
            if target_bytes is not None
            else JPEG_MIN_BYTES
        )
        max_bytes = (
            target_bytes + BENCHMARK_JPEG_TOLERANCE
            if target_bytes is not None
            else JPEG_MAX_BYTES
        )
        for size_index in range(count):
            records.append(
                _generate_jpeg(
                    output_dir
                    / "images"
                    / (
                        f"image_{image_index:04d}_{width}x{height}_{size_index:04d}.jpg"
                    ),
                    seed + image_index,
                    (width, height),
                    min_bytes=min_bytes,
                    max_bytes=max_bytes,
                )
            )
            image_index += 1
    encoded_hashes = {str(record["encoded_sha256"]) for record in records}
    decoded_hashes = {str(record["decoded_rgb_sha256"]) for record in records}
    if len(encoded_hashes) != unique_images or len(decoded_hashes) != unique_images:
        raise RuntimeError("generated image pool is not globally unique")

    shared_prompt, observed_text_isl = _calibrate_prompt(
        text_isl,
        lambda prompt: len(tokenizer(prompt, add_special_tokens=False).input_ids),
    )
    if observed_text_isl != text_isl:
        raise RuntimeError(
            f"raw prompt produced {observed_text_isl} tokens, expected {text_isl}"
        )
    prompts_by_size = {
        f"{width}x{height}": shared_prompt for width, height, _ in normalized_sizes
    }

    prompts_by_path: dict[str, str] = {}
    decoder_isls_by_size: dict[str, set[int]] = {
        f"{width}x{height}": set() for width, height, _ in normalized_sizes
    }
    for record in records:
        with Image.open(str(record["path"])) as encoded:
            image = encoded.convert("RGB")
        size_key = f"{record['width']}x{record['height']}"
        prompt = prompts_by_size[size_key]
        image_inputs = processor.image_processor(images=[image], return_tensors="pt")
        grid = image_inputs["image_grid_thw"][0]
        raw_patch_rows = int(grid.prod().item())
        merge_size = int(processor.image_processor.merge_size)
        observed_isl = _calculate_custom_isl_components(
            tokenizer, processor.image_processor, prompt, image
        )
        decoder_isls_by_size[size_key].add(observed_isl)
        record["grid_thw"] = [int(value) for value in grid.tolist()]
        record["raw_patch_rows"] = raw_patch_rows
        record["merged_visual_tokens"] = raw_patch_rows // merge_size**2
        prompts_by_path[str(record["path"])] = prompt

    schedule = _request_schedule(
        [str(record["path"]) for record in records], requests, seed
    )
    records_by_path = {str(record["path"]): record for record in records}
    rows = [
        {
            "session_id": f"request-{index:04d}",
            "image": image_path,
            "text": prompts_by_path[image_path],
        }
        for index, image_path in enumerate(schedule)
    ]
    input_name = f"image_custom_{requests}_textisl{text_isl}.jsonl"
    input_path = output_dir / input_name
    with input_path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, separators=(",", ":")) + "\n")

    sizes = [int(record["size_bytes"]) for record in records]
    occurrence_counts = Counter(schedule)
    requests_by_size = Counter(
        f"{records_by_path[path]['width']}x{records_by_path[path]['height']}"
        for path in schedule
    )
    size_manifest = [
        {
            "width": width,
            "height": height,
            "unique_images": count,
            "requests": requests_by_size[f"{width}x{height}"],
        }
        for width, height, count in normalized_sizes
    ]
    observed_decoder_isl_by_size: dict[str, int] = {}
    for size_key, observed in decoder_isls_by_size.items():
        if len(observed) != 1:
            raise RuntimeError(
                f"{size_key} images produced inconsistent decoder ISLs: {observed}"
            )
        observed_decoder_isl_by_size[size_key] = next(iter(observed))

    manifest: dict[str, Any] = {
        "concurrency": CONCURRENCY,
        "decoder_model": decoder_model,
        "encoder_model": encoder_model,
        "requests_per_concurrency": requests,
        "warmup_requests": 20,
        "unique_images": unique_images,
        "seed": seed,
        "target_osl": TARGET_OSL,
        "prompts_by_image_size": prompts_by_size,
        "image_size_counts": size_manifest,
        "encoding": {
            "format": "JPEG",
            "quality": 85,
            "subsampling": "4:2:0",
            "min_bytes": JPEG_MIN_BYTES,
            "max_bytes": JPEG_MAX_BYTES,
            "size_targets": {
                f"{width}x{height}": {
                    "target_bytes": target,
                    "min_bytes": target - BENCHMARK_JPEG_TOLERANCE,
                    "max_bytes": target + BENCHMARK_JPEG_TOLERANCE,
                }
                for (width, height), target in BENCHMARK_JPEG_TARGETS.items()
                if any(
                    (entry_width, entry_height) == (width, height)
                    for entry_width, entry_height, _ in normalized_sizes
                )
            },
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
        "observed_decoder_isl_by_image_size": observed_decoder_isl_by_size,
    }
    manifest.update(
        {
            "text_isl": text_isl,
            "prompt": shared_prompt,
            "prompt_sha256": hashlib.sha256(shared_prompt.encode("utf-8")).hexdigest(),
            "prompt_policy": (
                "one shared exact-token raw-text prompt plus one image per request"
            ),
        }
    )
    manifest_path = output_dir / "workload_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"workload={input_path.resolve()} images={unique_images} "
        f"requests={requests} sizes={size_manifest} text_isl={text_isl}"
    )
    return manifest_path


def validate_workload(
    root: Path,
    expected_image_size: int | None = None,
    expected_unique_images: int | None = None,
    expected_image_size_counts: tuple[tuple[int, int, int], ...] | None = None,
) -> dict[str, Any]:
    manifest_path = root / "workload_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "image_size_counts" in manifest:
        size_entries = manifest["image_size_counts"]
    else:
        decoded_image = manifest["decoded_image"]
        size_entries = [
            {
                "width": decoded_image["width"],
                "height": decoded_image["height"],
                "unique_images": manifest.get("unique_images", 1),
                "requests": manifest.get("requests_per_concurrency", 1),
            }
        ]
    manifest_size_counts = tuple(
        (int(entry["width"]), int(entry["height"]), int(entry["unique_images"]))
        for entry in size_entries
    )
    if any(min(entry) < 1 for entry in manifest_size_counts):
        raise AssertionError("manifest image dimensions and counts must be positive")
    if expected_image_size is not None and tuple(
        (width, height) for width, height, _ in manifest_size_counts
    ) != ((expected_image_size, expected_image_size),):
        raise AssertionError(
            "workload image size does not match requested "
            f"{expected_image_size}x{expected_image_size}"
        )
    if expected_image_size_counts is not None:
        expected = _normalize_image_size_counts(1, 1, expected_image_size_counts)
        if manifest_size_counts != expected:
            raise AssertionError(
                f"workload size counts {manifest_size_counts} do not match {expected}"
            )
    requests = int(manifest["requests_per_concurrency"])
    unique_images = int(manifest["unique_images"])
    if expected_unique_images is not None and unique_images != expected_unique_images:
        raise AssertionError(
            f"workload has {unique_images} unique images; requested "
            f"{expected_unique_images}"
        )
    text_isl = int(manifest["text_isl"])
    default_min_bytes = int(manifest["encoding"]["min_bytes"])
    default_max_bytes = int(manifest["encoding"]["max_bytes"])
    size_targets = manifest["encoding"].get("size_targets", {})
    input_path = Path(manifest["input"]["path"])
    rows = [
        json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines()
    ]
    if len(rows) != requests or manifest["input"]["sha256"] != _sha256(input_path):
        raise AssertionError("input JSONL count or hash mismatch")
    if len({row["session_id"] for row in rows}) != requests:
        raise AssertionError("session IDs must be unique")
    image_records = {str(record["path"]): record for record in manifest["images"]}
    image_paths = set(image_records)
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
    shared_prompt = str(manifest["prompt"])
    observed_text_isl = len(
        tokenizer(shared_prompt, add_special_tokens=False).input_ids
    )
    if observed_text_isl != text_isl:
        raise AssertionError(
            f"raw prompt has {observed_text_isl} tokens, expected {text_isl}"
        )
    if (
        manifest.get("prompt_sha256")
        != hashlib.sha256(shared_prompt.encode("utf-8")).hexdigest()
    ):
        raise AssertionError("prompt hash mismatch")
    prompts_by_size = {
        f"{width}x{height}": shared_prompt for width, height, _ in manifest_size_counts
    }
    if len({str(row["text"]) for row in rows}) != 1:
        raise AssertionError("text-ISL workload must use one shared prompt")
    expected_decoder_isls = {
        str(size): int(value)
        for size, value in manifest.get(
            "observed_decoder_isl_by_image_size", {}
        ).items()
    }
    actual_decoder_isls: dict[str, set[int]] = {
        f"{width}x{height}": set() for width, height, _ in manifest_size_counts
    }
    actual_size_counts: Counter[tuple[int, int]] = Counter()
    total_raw_patch_rows = 0
    total_merged_visual_tokens = 0
    for record in manifest["images"]:
        path = Path(record["path"])
        payload = path.read_bytes()
        size_key = f"{record['width']}x{record['height']}"
        size_target = size_targets.get(size_key, {})
        min_bytes = int(size_target.get("min_bytes", default_min_bytes))
        max_bytes = int(size_target.get("max_bytes", default_max_bytes))
        if not min_bytes <= len(payload) <= max_bytes:
            raise AssertionError(f"JPEG size out of range: {path}")
        encoded_hash = hashlib.sha256(payload).hexdigest()
        if encoded_hash != record["encoded_sha256"]:
            raise AssertionError(f"encoded hash mismatch: {path}")
        expected_dimensions = (int(record["width"]), int(record["height"]))
        with Image.open(path) as encoded:
            if encoded.format != "JPEG" or encoded.size != expected_dimensions:
                raise AssertionError(f"invalid JPEG shape or format: {path}")
            image = encoded.convert("RGB")
        decoded_hash = hashlib.sha256(image.tobytes()).hexdigest()
        if decoded_hash != record["decoded_rgb_sha256"]:
            raise AssertionError(f"decoded hash mismatch: {path}")
        size_key = f"{expected_dimensions[0]}x{expected_dimensions[1]}"
        prompt = str(prompts_by_size[size_key])
        observed_decoder_isl = _calculate_custom_isl_components(
            tokenizer, processor.image_processor, prompt, image
        )
        actual_decoder_isls[size_key].add(observed_decoder_isl)
        if observed_decoder_isl != expected_decoder_isls.get(size_key):
            raise AssertionError(f"decoder ISL mismatch: {path}")
        image_inputs = processor.image_processor(images=[image], return_tensors="pt")
        grid = image_inputs["image_grid_thw"][0]
        raw_patch_rows = int(grid.prod().item())
        merge_size = int(processor.image_processor.merge_size)
        merged_visual_tokens = raw_patch_rows // merge_size**2
        if "raw_patch_rows" in record and raw_patch_rows != int(
            record["raw_patch_rows"]
        ):
            raise AssertionError(f"raw patch count mismatch: {path}")
        if "merged_visual_tokens" in record and merged_visual_tokens != int(
            record["merged_visual_tokens"]
        ):
            raise AssertionError(f"merged visual token count mismatch: {path}")
        actual_size_counts[expected_dimensions] += 1
        total_raw_patch_rows += occurrence_counts[str(path)] * raw_patch_rows
        total_merged_visual_tokens += (
            occurrence_counts[str(path)] * merged_visual_tokens
        )
        encoded_hashes.add(encoded_hash)
        decoded_hashes.add(decoded_hash)
    if len(encoded_hashes) != unique_images or len(decoded_hashes) != unique_images:
        raise AssertionError("image uniqueness audit failed")

    expected_sizes = Counter(
        {(width, height): count for width, height, count in manifest_size_counts}
    )
    if actual_size_counts != expected_sizes:
        raise AssertionError(
            f"decoded image size counts {actual_size_counts} do not match "
            f"{expected_sizes}"
        )
    path_size = {
        path: (int(record["width"]), int(record["height"]))
        for path, record in image_records.items()
    }
    row_size_counts = Counter(path_size[str(row["image"])] for row in rows)
    expected_request_counts = Counter(
        {
            (int(entry["width"]), int(entry["height"])): int(entry["requests"])
            for entry in size_entries
        }
    )
    if row_size_counts != expected_request_counts:
        raise AssertionError("request image-size counts do not match manifest")
    for row in rows:
        width, height = path_size[str(row["image"])]
        if row["text"] != prompts_by_size[f"{width}x{height}"]:
            raise AssertionError("request prompt does not match its image size")

    observed_decoder_isl_by_size: dict[str, int] = {}
    for size_key, observed in actual_decoder_isls.items():
        if len(observed) != 1:
            raise AssertionError(
                f"{size_key} images produced inconsistent decoder ISLs: {observed}"
            )
        observed_decoder_isl_by_size[size_key] = next(iter(observed))

    result = {
        "manifest_sha256": _sha256(manifest_path),
        "input_sha256": _sha256(input_path),
        "requests": requests,
        "images": unique_images,
        "image_size_counts": [
            {"width": width, "height": height, "count": count}
            for width, height, count in manifest_size_counts
        ],
        "reuse_counts": actual_counts,
        "raw_patch_rows": total_raw_patch_rows,
        "merged_visual_tokens": total_merged_visual_tokens,
        "observed_decoder_isl_by_image_size": observed_decoder_isl_by_size,
    }
    result["text_isl"] = text_isl
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
    generate.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    generate.add_argument("--requests", type=int, default=REQUESTS)
    generate.add_argument("--seed", type=int, default=SEED)
    generate.add_argument(
        "--text-isl",
        type=int,
        default=TEXT_ISL,
        help="shared raw tokenizer length before the chat template and image",
    )
    generate.add_argument(
        "--image-size-count",
        action="append",
        type=_parse_image_size_count,
        help="repeatable WIDTHxHEIGHT:UNIQUE_IMAGES (for example 300x300:500)",
    )
    validate = subparsers.add_parser("validate")
    validate.add_argument("workload_dir", type=Path)
    validate.add_argument("--image-size", type=int)
    validate.add_argument("--unique-images", type=int)
    validate.add_argument(
        "--image-size-count",
        action="append",
        type=_parse_image_size_count,
    )
    args = parser.parse_args()
    if args.command == "generate":
        generate_workload(
            args.output_dir.resolve(),
            decoder_model=args.decoder_model,
            encoder_model=args.encoder_model,
            image_size=args.image_size,
            requests=args.requests,
            seed=args.seed,
            text_isl=args.text_isl,
            image_size_counts=(
                tuple(args.image_size_count) if args.image_size_count else None
            ),
        )
    else:
        validate_workload(
            args.workload_dir.resolve(),
            expected_image_size=args.image_size,
            expected_unique_images=args.unique_images,
            expected_image_size_counts=(
                tuple(args.image_size_count) if args.image_size_count else None
            ),
        )


if __name__ == "__main__":
    main()
