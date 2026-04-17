# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dynamo.vllm.multimodal_utils.hash_utils.

The hash preimage must include image geometry; otherwise two RGB images with
different (W, H) but equal pixel count produce identical cache keys. These
tests also pin the canonicalization contract (RGB uint8, C-contiguous) and
the on-disk preimage format via a known-digest stability anchor.
"""

import time

import blake3
import numpy as np
import pytest
from PIL import Image

from dynamo.vllm.multimodal_utils.hash_utils import compute_mm_uuids_from_images

pytestmark = [pytest.mark.pre_merge, pytest.mark.vllm]


# ---------------------------------------------------------------------------
# Regression: dimension-swap collision (TDD anchor)
# ---------------------------------------------------------------------------


def _dimension_swap_buf() -> bytes:
    buf = bytes(range(256)) * ((30 * 150 * 3) // 256 + 1)
    return buf[: 30 * 150 * 3]


def test_dimension_swap_no_collision_pil():
    """Two RGB PIL images sharing the same flat pixel buffer but with swapped
    (W, H) must hash to different UUIDs. PIL.Image.tobytes() emits raw pixel
    bytes with no geometry, so the preimage must carry dimensions explicitly.
    """
    buf = _dimension_swap_buf()
    wide = Image.frombytes("RGB", (150, 30), buf)
    tall = Image.frombytes("RGB", (30, 150), buf)

    assert wide.tobytes() == tall.tobytes(), (
        "Precondition: images must share raw pixel bytes for this test to "
        "exercise the geometry-in-preimage guarantee."
    )

    [wide_uuid] = compute_mm_uuids_from_images([wide])
    [tall_uuid] = compute_mm_uuids_from_images([tall])

    assert wide_uuid != tall_uuid


def test_dimension_swap_no_collision_ndarray():
    """Same as above but for the ndarray input path (NIXL / decoded pipeline)."""
    flat = np.arange(30 * 150 * 3, dtype=np.uint8) % 251
    wide = flat.reshape(30, 150, 3)
    tall = flat.reshape(150, 30, 3)

    assert wide.tobytes() == tall.tobytes()

    [wide_uuid] = compute_mm_uuids_from_images([wide])
    [tall_uuid] = compute_mm_uuids_from_images([tall])

    assert wide_uuid != tall_uuid


# ---------------------------------------------------------------------------
# Parity across pipelines
# ---------------------------------------------------------------------------


def test_pil_ndarray_equivalent_inputs_match():
    """A PIL image and its np.asarray() counterpart must produce identical
    UUIDs so URL-decode and NIXL-decode paths dedup the same logical image.
    """
    rng = np.random.default_rng(0xC0FFEE)
    arr = rng.integers(0, 256, size=(17, 23, 3), dtype=np.uint8)
    pil = Image.fromarray(arr, mode="RGB")

    [pil_uuid] = compute_mm_uuids_from_images([pil])
    [arr_uuid] = compute_mm_uuids_from_images([arr])

    assert pil_uuid == arr_uuid


# ---------------------------------------------------------------------------
# Canonicalization contract
# ---------------------------------------------------------------------------


def test_rejects_non_rgb_pil():
    img = Image.new("L", (8, 8))
    with pytest.raises(ValueError):
        compute_mm_uuids_from_images([img])


def test_rejects_non_rgb_pil_rgba():
    img = Image.new("RGBA", (8, 8))
    with pytest.raises(ValueError):
        compute_mm_uuids_from_images([img])


def test_rejects_wrong_dtype_ndarray():
    arr = np.zeros((8, 8, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        compute_mm_uuids_from_images([arr])


def test_rejects_wrong_shape_ndarray_2d():
    arr = np.zeros((8, 8), dtype=np.uint8)
    with pytest.raises(ValueError):
        compute_mm_uuids_from_images([arr])


def test_rejects_wrong_shape_ndarray_4ch():
    arr = np.zeros((8, 8, 4), dtype=np.uint8)
    with pytest.raises(ValueError):
        compute_mm_uuids_from_images([arr])


def test_rejects_bytes_input():
    with pytest.raises(TypeError):
        compute_mm_uuids_from_images([b"\x00" * (8 * 8 * 3)])


def test_non_contiguous_ndarray_coerced():
    """Non-C-contiguous ndarray views must still hash to the same value as
    an explicit contiguous copy of the same pixels.
    """
    rng = np.random.default_rng(42)
    contiguous = rng.integers(0, 256, size=(12, 20, 3), dtype=np.uint8)

    rgba = np.zeros((12, 20, 4), dtype=np.uint8)
    rgba[..., :3] = contiguous
    view = rgba[..., :3]
    assert not view.flags["C_CONTIGUOUS"]

    [view_uuid] = compute_mm_uuids_from_images([view])
    [contig_uuid] = compute_mm_uuids_from_images([contiguous])

    assert view_uuid == contig_uuid


# ---------------------------------------------------------------------------
# Stability anchor — pins the on-disk preimage format
# ---------------------------------------------------------------------------


def test_known_digest_stability():
    """A pinned 8x4 RGB gradient must hash to a fixed hex digest. If the
    preimage layout ever changes unintentionally, this test fails. If it is
    ever changed intentionally, bump the preimage version byte and update
    the pinned digest in the same commit.
    """
    h, w = 4, 8
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            arr[y, x] = (x * 16, y * 32, (x + y) * 8)

    [uuid] = compute_mm_uuids_from_images([arr])
    assert uuid == "1a53ddd0d1539154841e71befde56e9d90661e41b2256223f9ab9ed3fc7c02d5"


# ---------------------------------------------------------------------------
# Performance sanity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", [64, 512, 2048])
def test_hash_latency_no_regression(size):
    """Hardened path must not be more than ~1.5x slower than a raw blake3 of
    the same pixel buffer. Guards against accidental quadratic blowup (e.g.
    allocating header + pixels into a new buffer per image).
    """
    rng = np.random.default_rng(size)
    arr = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)

    trials = 3
    t0 = time.perf_counter()
    for _ in range(trials):
        blake3.blake3(arr.tobytes()).hexdigest()
    baseline = (time.perf_counter() - t0) / trials

    t0 = time.perf_counter()
    for _ in range(trials):
        compute_mm_uuids_from_images([arr])
    hardened = (time.perf_counter() - t0) / trials

    slack = 1.5
    floor = 0.001
    assert hardened <= max(baseline * slack, floor), (
        f"Hardened hash latency regressed: baseline={baseline:.4f}s "
        f"hardened={hardened:.4f}s (size={size}x{size})"
    )
