# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from benchmarks.multimodal.jsonl.generate_images import (
    JPEG_TARGET_MAX_BYTES,
    JPEG_TARGET_MIN_BYTES,
    generate_target_sized_jpeg_pool,
)

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


def test_target_sized_jpeg_pool_is_deterministic_and_unique(tmp_path: Path) -> None:
    first = generate_target_sized_jpeg_pool(3, tmp_path / "first", seed=42)
    second = generate_target_sized_jpeg_pool(3, tmp_path / "second", seed=42)

    assert [record["encoded_sha256"] for record in first] == [
        record["encoded_sha256"] for record in second
    ]
    assert len({record["encoded_sha256"] for record in first}) == 3
    assert len({record["decoded_rgb_sha256"] for record in first}) == 3
    for record in first:
        assert JPEG_TARGET_MIN_BYTES <= record["size_bytes"] <= JPEG_TARGET_MAX_BYTES
        with Image.open(record["path"]) as image:
            assert image.format == "JPEG"
            assert image.mode == "RGB"
            assert image.size == (500, 500)
