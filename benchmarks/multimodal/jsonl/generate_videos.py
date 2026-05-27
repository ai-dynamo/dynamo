# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for loading and sampling video URL pools."""

import json
import random
from pathlib import Path
from typing import Any


def _extract_video_ref(value: Any) -> str | None:
    """Extract one video reference from a manifest value.

    Accepted shapes are intentionally small and common:
      - "https://.../clip.mp4"
      - {"url": "..."}
      - {"video": "..."}
      - {"video_url": "..."} or {"video_url": {"url": "..."}}
    """
    if isinstance(value, str):
        return value.strip() or None
    if not isinstance(value, dict):
        return None

    for key in ("url", "video"):
        ref = value.get(key)
        if isinstance(ref, str) and ref.strip():
            return ref.strip()

    video_url = value.get("video_url")
    if isinstance(video_url, str) and video_url.strip():
        return video_url.strip()
    if isinstance(video_url, dict):
        ref = video_url.get("url")
        if isinstance(ref, str) and ref.strip():
            return ref.strip()

    return None


def _iter_manifest_records(path: Path) -> list[Any]:
    text = path.read_text().strip()
    if not text:
        return []

    if path.suffix.lower() == ".json":
        data = json.loads(text)
        if isinstance(data, dict):
            for key in ("videos", "data", "items"):
                if isinstance(data.get(key), list):
                    return data[key]
            return [data]
        if isinstance(data, list):
            return data
        raise ValueError(f"Unsupported JSON manifest root type: {type(data).__name__}")

    records: list[Any] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            records.append(line)
        if records[-1] is None:
            raise ValueError(f"Invalid empty record at {path}:{line_no}")
    return records


def load_video_pool(
    py_rng: random.Random,
    manifest_path: Path,
    pool_size: int,
) -> list[str]:
    """Load up to pool_size unique video references from a text/json/jsonl manifest."""
    records = _iter_manifest_records(manifest_path)
    refs: list[str] = []
    seen: set[str] = set()
    for record in records:
        ref = _extract_video_ref(record)
        if ref and ref not in seen:
            refs.append(ref)
            seen.add(ref)

    if pool_size > len(refs):
        raise RuntimeError(
            f"--videos-pool ({pool_size}) exceeds available videos in "
            f"{manifest_path} ({len(refs)}). Reduce --videos-pool or add videos."
        )

    py_rng.shuffle(refs)
    pool = refs[:pool_size]
    print(
        f"  {pool_size} video URLs sampled from {manifest_path} ({len(refs)} available)"
    )
    return pool


def sample_video_slots(
    py_rng: random.Random,
    pool: list[str],
    num_requests: int,
    videos_per_request: int,
) -> list[str]:
    """Sample video slots from a fixed pool, no duplicates within each request.

    Every video in the pool is guaranteed to appear at least once.
    """
    pool_size = len(pool)
    total_slots = num_requests * videos_per_request
    assert (
        pool_size >= videos_per_request
    ), f"videos-pool ({pool_size}) must be >= videos-per-request ({videos_per_request})"
    assert total_slots >= pool_size, (
        f"total slots ({num_requests}×{videos_per_request}={total_slots}) < "
        f"videos-pool ({pool_size}). Increase --num-requests or --videos-per-request, "
        f"or reduce --videos-pool."
    )

    # Round-robin every pool video into requests so each appears at least once
    shuffled = list(pool)
    py_rng.shuffle(shuffled)
    requests: list[list[str]] = [[] for _ in range(num_requests)]
    for i, video in enumerate(shuffled):
        requests[i % num_requests].append(video)

    # Fill remaining slots with random pool samples (no intra-request duplicates)
    for req in requests:
        remaining = videos_per_request - len(req)
        if remaining > 0:
            used = set(req)
            available = [video for video in pool if video not in used]
            req.extend(py_rng.sample(available, remaining))
        py_rng.shuffle(req)

    slot_refs = [video for req in requests for video in req]
    num_unique = len(set(slot_refs))
    print(
        f"Generated {total_slots} video slots from pool of {pool_size}: "
        f"{num_unique} unique in use, "
        f"{total_slots - num_unique} duplicate references "
        f"({(total_slots - num_unique) / total_slots:.1%} reuse)"
    )
    return slot_refs
