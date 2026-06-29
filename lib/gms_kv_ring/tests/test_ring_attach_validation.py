# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import struct

import pytest


@pytest.mark.parametrize(
    "module_name",
    ["gms_kv_ring.common.evict_ring", "gms_kv_ring.common.restore_ring"],
)
def test_attach_rejects_truncated_header(tmp_path, module_name):
    ring = importlib.import_module(module_name)
    path = tmp_path / "truncated.ring"
    path.write_bytes(b"short")

    with pytest.raises(ValueError, match="too small"):
        ring.attach_reader(str(path))


@pytest.mark.parametrize(
    ("offset", "value", "message"),
    [
        (4, 99, "version mismatch"),
        (8, 3, "power-of-2"),
        (12, 128, "record_size mismatch"),
    ],
)
@pytest.mark.parametrize(
    "module_name",
    ["gms_kv_ring.common.evict_ring", "gms_kv_ring.common.restore_ring"],
)
def test_attach_rejects_invalid_header_field(
    tmp_path,
    module_name,
    offset,
    value,
    message,
):
    ring = importlib.import_module(module_name)
    path = tmp_path / "invalid.ring"
    writer = ring.create_ring(str(path), capacity=8)
    writer.close()
    with path.open("r+b") as file:
        file.seek(offset)
        file.write(struct.pack("<I", value))

    with pytest.raises(ValueError, match=message):
        ring.attach_reader(str(path))
