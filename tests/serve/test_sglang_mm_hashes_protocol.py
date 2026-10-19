# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pin sglang's MM-routing wire-protocol contracts against drift.

These fast unit tests import the installed sglang module and verify two
contracts that the dynamo Rust frontend depends on for MM-aware KV
routing:

  1. The pad tokens that sglang writes for a caller-supplied hash.
     Releases before sgl-project/sglang#43043 write one pad,
     ``MM_PAD_SHIFT_VALUE + (hash % (1 << 30))``, which is
     ``dynamo_kv_router::protocols::pad_value_for_mm_hash`` (Rust unit
     ``pad_value_matches_sglang_protocol``). Newer releases write the four
     wide pads of ``_compute_pad_values``, repeated over each placeholder
     run. The worker then publishes ``sglang_mm_pad_scheme: "wide_v1"`` and
     the frontend writes the same pads (Rust unit
     ``sglang_wide_pads_match_sglang_vectors`` pins the same vectors as
     this test). A sglang change to either scheme fails here at PR time.

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


# Raw sglang wide pads for a caller hash, from ``set_hash(h).padding_values()``.
WIDE_PAD_VECTORS = {
    0x7BE956BE787A1C98: (
        0x2934B765019440FF,
        0x0E108BD9B4491EAD,
        0x16CD70C642BAAC32,
        0x2926960A32A678AB,
    ),
    0x0123456789ABCDEF: (
        0x00539BEE0B640CC4,
        0x0F7A1566ABE43794,
        0x05A4E8716D5E2999,
        0x1D25C7F3F8C13FEA,
    ),
}


def test_sglang_pad_values_match_dynamo() -> None:
    from sglang.srt.managers import schedule_batch

    assert schedule_batch.MM_PAD_SHIFT_VALUE == 1_000_000, (
        f"sglang MM_PAD_SHIFT_VALUE changed to {schedule_batch.MM_PAD_SHIFT_VALUE}; "
        "dynamo's MM pads in dynamo_kv_router::protocols and "
        "lib/llm/src/preprocessor/mm_routing must be updated in lockstep or "
        "MM-aware routing will misroute."
    )
    if not hasattr(schedule_batch, "_compute_pad_values"):
        compute_pad_value = schedule_batch._compute_pad_value
        assert compute_pad_value(0) == 1_000_000
        fits = (1 << 30) - 1
        assert compute_pad_value(fits) == 1_000_000 + fits
        overflow = (1 << 30) | 0xCAFE
        assert compute_pad_value(overflow) == 1_000_000 + 0xCAFE, (
            "high bits above the 30-bit mask must be discarded; sglang's "
            "formula switched away from `hash % (1 << 30)` if this fails"
        )
        return

    for mm_hash, expected in WIDE_PAD_VECTORS.items():
        item = schedule_batch.MultimodalDataItem(modality=schedule_batch.Modality.IMAGE)
        # The worker forwards Dynamo's hash as 16 hex digits.
        item.set_hash(int(f"{mm_hash:016x}", 16))
        assert item.padding_values() == expected, (
            "sglang's wide pad derivation changed; update Dynamo's "
            "sglang_wide_pads and its pad scheme version"
        )
        assert item.padding_sequence(6) == [
            *expected,
            *expected[:2],
        ], "sglang no longer repeats the pads in order over a placeholder run"


def test_sglang_generate_req_input_has_mm_hashes_field() -> None:
    from sglang.srt.managers.io_struct import GenerateReqInput

    fields = {f.name for f in dataclasses.fields(GenerateReqInput)}
    assert "mm_hashes" in fields, (
        "GenerateReqInput.mm_hashes is missing. The dynamo sglang image "
        "was built with an older SGLang than v0.5.13, or upstream renamed "
        "the field. Without it, sglang ignores caller-supplied hashes and "
        "MM-aware KV routing silently degrades to text-prefix fallback."
    )
