# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for generating and sampling image pools."""

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image


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
