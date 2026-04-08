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
    offset: int = 0,
) -> list[str]:
    """Generate pool_size random PNG files starting at img_{offset:04d} and return their paths."""
    image_dir.mkdir(parents=True, exist_ok=True)
    pool: list[str] = []
    for idx in range(offset, offset + pool_size):
        path = image_dir / f"img_{idx:04d}.png"
        pixels = np_rng.integers(0, 256, (*image_size, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(path)
        pool.append(str(path.resolve()))
    print(
        f"  {pool_size} unique {image_size[0]}x{image_size[1]} images saved to {image_dir} "
        f"(offset={offset}, range img_{offset:04d}…img_{offset+pool_size-1:04d})"
    )
    return pool


def generate_image_pool_http(
    py_rng: random.Random,
    pool_size: int,
    coco_annotations: Path,
    offset: int = 0,
) -> list[str]:
    """Pick pool_size unique COCO test2017 URLs starting after offset URLs in the shuffled list."""
    with open(coco_annotations) as f:
        data = json.load(f)
    all_urls = [img["coco_url"] for img in data["images"]]
    if offset + pool_size > len(all_urls):
        raise RuntimeError(
            f"--image-offset ({offset}) + --images-pool ({pool_size}) = {offset + pool_size} "
            f"exceeds available COCO images ({len(all_urls)}). "
            f"Reduce --images-pool or --image-offset."
        )
    py_rng.shuffle(all_urls)
    pool = all_urls[offset : offset + pool_size]
    print(
        f"  {pool_size} URLs sampled from {coco_annotations.name} ({len(all_urls)} available, "
        f"offset={offset})"
    )
    return pool


def sample_slots(
    py_rng: random.Random,
    pool: list[str],
    num_requests: int,
    images_per_request: int,
) -> list[str]:
    """Sample image slots from a fixed pool, no duplicates within each request."""
    assert (
        len(pool) >= images_per_request
    ), f"images-pool ({len(pool)}) must be >= images-per-request ({images_per_request})"
    total_slots = num_requests * images_per_request
    slot_refs: list[str] = []
    for _ in range(num_requests):
        slot_refs.extend(py_rng.sample(pool, images_per_request))

    num_unique = len(set(slot_refs))
    print(
        f"Generated {total_slots} image slots from pool of {len(pool)}: "
        f"{num_unique} unique in use, "
        f"{total_slots - num_unique} duplicate references "
        f"({(total_slots - num_unique) / total_slots:.1%} reuse)"
    )
    return slot_refs
