# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dynamo.vllm.multimodal_utils.hash_utils.

The hash preimage must include image geometry; otherwise two RGB images with
different (W, H) but equal pixel count produce identical cache keys. These
tests also pin the canonicalization contract (RGB uint8, C-contiguous) and
the on-disk preimage format via a known-digest stability anchor.
"""

import numpy as np
import pytest
from PIL import Image

from dynamo.vllm.multimodal_utils.hash_utils import compute_mm_uuids_from_images

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]


# ---------------------------------------------------------------------------
# Regression: dimension-swap collision (TDD anchor)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make_image",
    [
        pytest.param(
            lambda h, w, buf: Image.frombytes("RGB", (w, h), buf),
            id="pil",
        ),
        pytest.param(
            lambda h, w, buf: np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3),
            id="ndarray",
        ),
    ],
)
def test_dimension_swap_no_collision(make_image):
    """Two RGB images sharing the same flat pixel buffer but with swapped
    (W, H) must hash to different UUIDs. Raw pixel bytes carry no geometry,
    so the preimage must include dimensions explicitly. Covers both the PIL
    input path (URL decode) and the ndarray path (NIXL / Rust decoder).
    """
    buf = bytes(range(256)) * ((30 * 150 * 3) // 256 + 1)
    buf = buf[: 30 * 150 * 3]

    wide = make_image(30, 150, buf)
    tall = make_image(150, 30, buf)

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


@pytest.mark.parametrize(
    "bad_input, exc",
    [
        pytest.param(
            lambda: Image.new("L", (8, 8)),
            ValueError,
            id="pil_mode_L",
        ),
        pytest.param(
            lambda: Image.new("RGBA", (8, 8)),
            ValueError,
            id="pil_mode_RGBA",
        ),
        pytest.param(
            lambda: np.zeros((8, 8, 3), dtype=np.float32),
            ValueError,
            id="ndarray_dtype_float32",
        ),
        pytest.param(
            lambda: np.zeros((8, 8), dtype=np.uint8),
            ValueError,
            id="ndarray_shape_2d",
        ),
        pytest.param(
            lambda: np.zeros((8, 8, 4), dtype=np.uint8),
            ValueError,
            id="ndarray_shape_4ch",
        ),
        pytest.param(
            lambda: b"\x00" * (8 * 8 * 3),
            TypeError,
            id="bytes",
        ),
    ],
)
def test_rejects_invalid_input(bad_input, exc):
    """Inputs outside the RGB uint8 (H, W, 3) contract must raise before any
    hashing work — loud failure beats silent collision.
    """
    with pytest.raises(exc):
        compute_mm_uuids_from_images([bad_input()])


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
