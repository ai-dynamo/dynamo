# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys

import pytest

from tests.serve.multimodal_profiles import sglang as sglang_profiles
from tests.serve.multimodal_profiles.sglang import (
    SGLangMmRoutingPayload,
    sglang_source_has_wide_mm_pad_values,
)

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]

# Trimmed from SGLang 0.5.23 sglang/srt/managers/schedule_batch.py.
WIDE_PAD_SOURCE = """
import msgspec

class MultimodalDataItem(msgspec.Struct, kw_only=True, dict=True, array_like=True):
    hash: int = None
    pad_value: Optional[int] = None
    # pad_value remains the first sentinel for scalar-padding consumers
    pad_values: Optional[Tuple[int, ...]] = None
"""

# Scalar-pad builds (0.5.22) mention pad_values outside the struct fields.
SCALAR_PAD_SOURCE = """
# This ensures pad_values don't overlap with valid text token IDs.
MSG = "MM pad_values may overlap with valid token IDs."

class MultimodalDataItem(msgspec.Struct, kw_only=True, dict=True, array_like=True):
    hash: int = None
    pad_value: Optional[int] = None

    def set_pad_values(self, pad_values):
        self.pad_values = pad_values
"""


def _write(tmp_path, text):
    path = tmp_path / "schedule_batch.py"
    path.write_text(text)
    return path


def _routing_payload() -> SGLangMmRoutingPayload:
    return SGLangMmRoutingPayload(
        body={}, repeat_count=3, expected_response=[], min_avg_kv_hit_rate=0.9
    )


def test_wide_pad_source_is_detected(tmp_path):
    assert sglang_source_has_wide_mm_pad_values(_write(tmp_path, WIDE_PAD_SOURCE))


def test_scalar_pad_source_is_not_wide(tmp_path):
    assert not sglang_source_has_wide_mm_pad_values(_write(tmp_path, SCALAR_PAD_SOURCE))


def test_missing_source_raises(tmp_path):
    with pytest.raises(RuntimeError, match="cannot read SGLang source"):
        sglang_source_has_wide_mm_pad_values(tmp_path / "schedule_batch.py")


def test_unparseable_source_raises(tmp_path):
    with pytest.raises(RuntimeError, match="cannot read SGLang source"):
        sglang_source_has_wide_mm_pad_values(_write(tmp_path, "class (:\n"))


def test_source_without_struct_raises(tmp_path):
    with pytest.raises(RuntimeError, match="MultimodalDataItem not found"):
        sglang_source_has_wide_mm_pad_values(_write(tmp_path, "x = 1\n"))


def test_sglang_not_installed_raises(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match="sglang is not installed"):
        sglang_source_has_wide_mm_pad_values()


@pytest.mark.sglang
def test_installed_sglang_source_is_found():
    assert isinstance(sglang_source_has_wide_mm_pad_values(), bool)


@pytest.mark.parametrize(
    ("source", "expected_floor"), [(WIDE_PAD_SOURCE, 0.0), (SCALAR_PAD_SOURCE, 0.9)]
)
def test_bind_environment_floor_ignores_runtime_import_failure(
    monkeypatch, tmp_path, source, expected_floor
):
    # The runtime module failing to import must not change the decision.
    monkeypatch.setitem(sys.modules, "sglang.srt.managers.schedule_batch", None)
    path = _write(tmp_path, source)
    monkeypatch.setattr(
        sglang_profiles,
        "sglang_source_has_wide_mm_pad_values",
        lambda: sglang_source_has_wide_mm_pad_values(path),
    )
    payload = _routing_payload()
    payload.bind_environment({})
    assert payload.min_avg_kv_hit_rate == expected_floor
