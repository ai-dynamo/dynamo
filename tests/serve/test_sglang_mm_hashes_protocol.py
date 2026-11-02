# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pin sglang's MM-routing wire-protocol contracts against drift.

These fast unit tests import the installed sglang module and verify two
contracts that the dynamo Rust frontend depends on for MM-aware KV
routing:

  1. ``MultimodalDataItem`` exposes ``pad_values``, and every image pad
     is at or above ``MM_PAD_SHIFT_VALUE == 1_000_000``. Dynamo detects
     that field to enable text-prefix routing on wide-token builds. Its
     KV event normalizer excludes blocks containing tokens at or above
     ``dynamo_kv_router::protocols::MM_PAD_SHIFT_VALUE`` and their
     descendants, without changing SGLang's cache identity.

  2. ``GenerateReqInput.mm_hashes`` field is present. Added by
     sgl-project/sglang#25300 and upstream in SGLang v0.5.13. A dynamo sglang
     image built with older SGLang — or a future upstream rename — would
     silently break MM-aware routing (workers recompute hashes internally,
     diverging from the router's pad_value substitution).
"""
from __future__ import annotations

import dataclasses

import pytest

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.sglang,
    pytest.mark.unit,
    pytest.mark.gpu_0,
]


def test_sglang_wide_pads_stay_above_router_exclusion_threshold() -> None:
    from sglang.srt.managers.schedule_batch import (
        MM_PAD_SHIFT_VALUE,
        Modality,
        MultimodalDataItem,
    )

    assert MM_PAD_SHIFT_VALUE == 1_000_000, (
        f"sglang MM_PAD_SHIFT_VALUE changed to {MM_PAD_SHIFT_VALUE}; the KV "
        "event normalizer excludes tokens at or above "
        "dynamo_kv_router::protocols::MM_PAD_SHIFT_VALUE."
    )
    assert "pad_values" in MultimodalDataItem.__struct_fields__
    item = MultimodalDataItem(modality=Modality.IMAGE, offsets=[(1, 5)])
    item.set_hash(0x9F3A6C1D2E4B5A78)
    assert min(item.padding_values()) >= MM_PAD_SHIFT_VALUE


def test_sglang_generate_req_input_has_mm_hashes_field() -> None:
    from sglang.srt.managers.io_struct import GenerateReqInput

    fields = {f.name for f in dataclasses.fields(GenerateReqInput)}
    assert "mm_hashes" in fields, (
        "GenerateReqInput.mm_hashes is missing. The dynamo sglang image "
        "was built with an older SGLang than v0.5.13, or upstream renamed "
        "the field. Without it, sglang ignores caller-supplied hashes and "
        "MM-aware KV routing silently degrades to text-prefix fallback."
    )
