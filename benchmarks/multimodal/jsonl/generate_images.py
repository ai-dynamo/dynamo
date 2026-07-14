# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for generating and sampling image pools."""

import hashlib
import io
import json
import random
import uuid as _uuid
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

JPEG_TARGET_MIN_BYTES = 50 * 1024
JPEG_TARGET_MAX_BYTES = 60 * 1024


def compute_image_uuid(ref: str) -> str:
    """Stable UUID for an image reference (path or URL).

    Used as the `uuid` field on chat-completion `image_url` content parts —
    a vLLM extension to the OpenAI-compat schema (see vLLM's
    `multi_modal_uuids` / `mm_processor_cache`). Same `ref` → same UUID
    across runs, so the server's processor cache survives benchmark
    restarts.

    Returns a hyphenated UUID string. Dynamo's chat parser enforces strict
    UUID format at the wire boundary, so this MUST be a valid UUID — we
    use UUIDv5 in the OID namespace for determinism.
    """
    return str(_uuid.uuid5(_uuid.NAMESPACE_OID, ref))


def generate_image_pool_base64(
    np_rng: np.random.Generator,
    pool_size: int,
    image_dir: Path,
    image_size: tuple[int, int] = (512, 512),
) -> list[str]:
    """Generate pool_size random PNG files and return their paths."""
    image_dir.mkdir(parents=True, exist_ok=True)
    pool: list[str] = []
    for idx in range(pool_size):
        path = image_dir / f"img_{idx:04d}.png"
        pixels = np_rng.integers(0, 256, (*image_size, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(path)
        pool.append(str(path.resolve()))
    print(
        f"  {pool_size} unique {image_size[0]}x{image_size[1]} images saved to {image_dir}"
    )
    return pool


def _encode_resampled_noise_jpeg(
    noise: Image.Image,
    texture_side: int,
    image_size: tuple[int, int],
    quality: int,
) -> bytes:
    """Encode deterministic, compressible noise at a fixed JPEG quality."""
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


def generate_target_sized_jpeg(
    np_rng: np.random.Generator,
    path: Path,
    image_size: tuple[int, int] = (500, 500),
    quality: int = 85,
    min_bytes: int = JPEG_TARGET_MIN_BYTES,
    max_bytes: int = JPEG_TARGET_MAX_BYTES,
) -> dict[str, Any]:
    """Write one deterministic JPEG whose encoded size is within a byte range.

    JPEG quality remains fixed. The generator changes only the resolution of a
    seeded noise texture, which controls compressibility without changing the
    decoded image dimensions. A 180-pixel texture normally lands near 55 KiB;
    binary search is used only when an encoder/version produces a result outside
    the requested range.
    """
    if min_bytes <= 0 or max_bytes < min_bytes:
        raise ValueError("expected 0 < min_bytes <= max_bytes")
    if not 1 <= quality <= 100:
        raise ValueError("quality must be between 1 and 100")

    width, height = image_size
    pixels = np_rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    noise = Image.fromarray(pixels)
    target_bytes = (min_bytes + max_bytes) // 2

    candidates: list[tuple[int, bytes]] = []

    def encode(texture_side: int) -> bytes:
        payload = _encode_resampled_noise_jpeg(noise, texture_side, image_size, quality)
        candidates.append((texture_side, payload))
        return payload

    payload = encode(min(180, width, height))
    if not min_bytes <= len(payload) <= max_bytes:
        lower = 8
        upper = min(width, height)
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
        "width": width,
        "height": height,
        "size_bytes": len(payload),
        "jpeg_quality": quality,
        "texture_side": texture_side,
        "encoded_sha256": hashlib.sha256(payload).hexdigest(),
        "decoded_rgb_sha256": decoded_hash,
    }


def generate_target_sized_jpeg_pool(
    pool_size: int,
    image_dir: Path,
    seed: int,
    image_size: tuple[int, int] = (500, 500),
    quality: int = 85,
    min_bytes: int = JPEG_TARGET_MIN_BYTES,
    max_bytes: int = JPEG_TARGET_MAX_BYTES,
    start_index: int = 0,
) -> list[dict[str, Any]]:
    """Generate a deterministic pool of unique target-sized JPEGs."""
    image_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    encoded_hashes: set[str] = set()
    decoded_hashes: set[str] = set()
    for offset in range(pool_size):
        index = start_index + offset
        record = generate_target_sized_jpeg(
            np.random.default_rng(seed + index),
            image_dir / f"image_{index:04d}_{image_size[0]}x{image_size[1]}.jpg",
            image_size=image_size,
            quality=quality,
            min_bytes=min_bytes,
            max_bytes=max_bytes,
        )
        if record["encoded_sha256"] in encoded_hashes:
            raise RuntimeError(f"duplicate encoded JPEG at {record['path']}")
        if record["decoded_rgb_sha256"] in decoded_hashes:
            raise RuntimeError(f"duplicate decoded RGB image at {record['path']}")
        encoded_hashes.add(record["encoded_sha256"])
        decoded_hashes.add(record["decoded_rgb_sha256"])
        records.append(record)
    return records


def generate_image_pool_http(
    py_rng: random.Random,
    pool_size: int,
    coco_annotations: Path,
) -> list[str]:
    """Pick pool_size unique COCO test2017 URLs."""
    with open(coco_annotations) as f:
        data = json.load(f)
    all_urls = [img["coco_url"] for img in data["images"]]
    if pool_size > len(all_urls):
        raise RuntimeError(
            f"--images-pool ({pool_size}) exceeds available COCO images ({len(all_urls)}). "
            f"Reduce --images-pool."
        )
    py_rng.shuffle(all_urls)
    pool = all_urls[:pool_size]
    print(
        f"  {pool_size} URLs sampled from {coco_annotations.name} ({len(all_urls)} available)"
    )
    return pool


def sample_slots(
    py_rng: random.Random,
    pool: list[str],
    num_requests: int,
    images_per_request: int,
) -> list[str]:
    """Sample image slots from a fixed pool, no duplicates within each request.

    Every image in the pool is guaranteed to appear at least once.
    """
    pool_size = len(pool)
    total_slots = num_requests * images_per_request
    assert (
        pool_size >= images_per_request
    ), f"images-pool ({pool_size}) must be >= images-per-request ({images_per_request})"
    assert total_slots >= pool_size, (
        f"total slots ({num_requests}×{images_per_request}={total_slots}) < "
        f"images-pool ({pool_size}). Increase --num-requests or --images-per-request, "
        f"or reduce --images-pool."
    )

    # Round-robin every pool image into requests so each appears at least once
    shuffled = list(pool)
    py_rng.shuffle(shuffled)
    requests: list[list[str]] = [[] for _ in range(num_requests)]
    for i, img in enumerate(shuffled):
        requests[i % num_requests].append(img)

    # Fill remaining slots with random pool samples (no intra-request duplicates)
    for req in requests:
        remaining = images_per_request - len(req)
        if remaining > 0:
            used = set(req)
            available = [img for img in pool if img not in used]
            req.extend(py_rng.sample(available, remaining))
        py_rng.shuffle(req)

    slot_refs = [img for req in requests for img in req]
    num_unique = len(set(slot_refs))
    print(
        f"Generated {total_slots} image slots from pool of {pool_size}: "
        f"{num_unique} unique in use, "
        f"{total_slots - num_unique} duplicate references "
        f"({(total_slots - num_unique) / total_slots:.1%} reuse)"
    )
    return slot_refs
