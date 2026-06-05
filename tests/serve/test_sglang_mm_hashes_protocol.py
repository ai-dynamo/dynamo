# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pin sglang's MM-routing wire-protocol contracts against drift.

These fast unit tests import the installed sglang module and verify two
contracts that the dynamo Rust frontend depends on for MM-aware KV
routing:

  1. ``sglang.srt.managers.schedule_batch._compute_pad_value(hash)`` must
     equal ``MM_PAD_SHIFT_VALUE + (hash % (1 << 30))`` with
     ``MM_PAD_SHIFT_VALUE == 1_000_000``. Mirrored in Rust at
     ``lib/llm/src/preprocessor.rs::pad_value_for_sglang``. Pinned by
     the Rust unit ``mm_pad_value_matches_sglang_protocol`` on our side;
     this test pins the same constants on sglang's side so a future
     sglang bump that shifts the formula fails-closed at PR time.

  2. ``GenerateReqInput.mm_hashes`` field is present. Added by
     sgl-project/sglang#25300 and currently shipped via the vendored
     patch at ``container/deps/sglang/patches/<ver>/``. A dynamo sglang
     image built without the patch — or a future upstream rename —
     would silently break MM-aware routing (workers recompute hashes
     internally, diverging from the router's pad_value substitution).
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


def test_sglang_pad_value_formula_unchanged() -> None:
    from sglang.srt.managers.schedule_batch import (
        MM_PAD_SHIFT_VALUE,
        _compute_pad_value,
    )

    assert MM_PAD_SHIFT_VALUE == 1_000_000, (
        f"sglang MM_PAD_SHIFT_VALUE changed to {MM_PAD_SHIFT_VALUE}; "
        "dynamo's pad_value_for_sglang in lib/llm/src/preprocessor.rs "
        "must be updated in lockstep or MM-aware routing will misroute."
    )
    assert _compute_pad_value(0) == 1_000_000
    fits = (1 << 30) - 1
    assert _compute_pad_value(fits) == 1_000_000 + fits
    overflow = (1 << 30) | 0xCAFE
    assert _compute_pad_value(overflow) == 1_000_000 + 0xCAFE, (
        "high bits above the 30-bit mask must be discarded; sglang's "
        "formula switched away from `hash % (1 << 30)` if this fails"
    )


def test_sglang_generate_req_input_has_mm_hashes_field() -> None:
    from sglang.srt.managers.io_struct import GenerateReqInput

    fields = {f.name for f in dataclasses.fields(GenerateReqInput)}
    assert "mm_hashes" in fields, (
        "GenerateReqInput.mm_hashes is missing. The dynamo sglang image "
        "was built without the vendored sgl-project/sglang#25300 patch "
        "(see container/deps/sglang/patches/), or upstream renamed the "
        "field. Without it, sglang ignores caller-supplied hashes and "
        "MM-aware KV routing silently degrades to text-prefix fallback."
    )
