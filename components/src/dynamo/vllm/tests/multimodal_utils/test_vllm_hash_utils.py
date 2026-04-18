# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for hash_utils — regression coverage for the
dimension-blind cache poisoning collision reported by Dem0."""

import blake3
import numpy as np
import pytest
from PIL import Image

from dynamo.vllm.multimodal_utils.hash_utils import (
    compute_mm_uuids_from_images,
    image_to_bytes,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]


def _build_tobytes_collision() -> tuple[Image.Image, Image.Image]:
    a = 30
    h1, w1 = 5 * a, 1 * a
    h2, w2 = 1 * a, 5 * a
    n = h1 * w1

    target_a = np.full((h1, w1), 255, dtype=np.uint8)
    target_b = np.full((h2, w2), 255, dtype=np.uint8)
    target_a[10:140, 5:25] = 0
    target_b[5:25, 10:140] = 0

    pixels = np.full(n, 255, dtype=np.uint8)
    for i in range(n):
        val_a = target_a[i // w1, i % w1]
        val_b = target_b[i // w2, i % w2]
        pixels[i] = min(val_a, val_b)

    img_a = Image.fromarray(pixels.reshape(h1, w1), "L").convert("RGB")
    img_b = Image.fromarray(pixels.reshape(h2, w2), "L").convert("RGB")
    return img_a, img_b


class TestCollisionResistance:
    def test_tobytes_collision_produces_distinct_hashes(self):
        img_a, img_b = _build_tobytes_collision()
        assert img_a.size != img_b.size
        assert img_a.tobytes() == img_b.tobytes()

        uuid_a, uuid_b = compute_mm_uuids_from_images([img_a, img_b])
        assert uuid_a != uuid_b

    def test_ndarray_shape_sensitive(self):
        flat = np.arange(12, dtype=np.uint8)
        arr_a = flat.reshape(3, 4)
        arr_b = flat.reshape(4, 3)
        assert arr_a.tobytes() == arr_b.tobytes()

        uuid_a, uuid_b = compute_mm_uuids_from_images([arr_a, arr_b])
        assert uuid_a != uuid_b

    def test_mode_sensitive(self):
        pixels = np.array([[10, 20], [30, 40]], dtype=np.uint8)
        img_l = Image.fromarray(pixels, "L")
        img_rgb = Image.merge("RGB", (img_l, img_l, img_l))

        uuid_l, uuid_rgb = compute_mm_uuids_from_images([img_l, img_rgb])
        assert uuid_l != uuid_rgb


class TestStability:
    def test_bytes_passthrough_unchanged(self):
        payload = b"opaque-precomputed-bytes"
        assert image_to_bytes(payload) is payload
        [uuid] = compute_mm_uuids_from_images([payload])
        assert uuid == blake3.blake3(payload).hexdigest()

    def test_same_image_stable_hash(self):
        img = Image.new("RGB", (8, 8), color=(1, 2, 3))
        [u1] = compute_mm_uuids_from_images([img])
        [u2] = compute_mm_uuids_from_images([img])
        assert u1 == u2


class TestUnsupportedType:
    def test_raises_type_error(self):
        with pytest.raises(TypeError):
            image_to_bytes(12345)
