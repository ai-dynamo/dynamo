# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pin sglang's MM-routing wire-protocol contracts against drift.

These fast unit tests import the installed sglang module and verify two
contracts that the dynamo Rust frontend depends on for MM-aware KV
routing:

  1. The legacy scalar pad value must equal
     ``MM_PAD_SHIFT_VALUE + (hash % (1 << 30))`` with
     ``MM_PAD_SHIFT_VALUE == 1_000_000``. Mirrored in Rust at
     ``dynamo_kv_router::protocols::pad_value_for_mm_hash`` (shared by the
     frontend and the kv-router's vLLM-event normalization). Pinned by
     the Rust unit ``pad_value_matches_sglang_protocol`` on our side;
     this test pins the same constants on sglang's side so a future
     sglang bump that shifts the formula fails-closed at PR time. The Kimi
     fork instead uses wide identity padding; pin that algorithm separately
     and verify Dynamo does not forward caller hashes to Kimi's processor.
     Wide engine sentinels are not Dynamo's canonical routing pad values.

  2. ``GenerateReqInput.mm_hashes`` field is present. Added by
     sgl-project/sglang#25300 and upstream in SGLang v0.5.13. A dynamo sglang
     image built with older SGLang — or a future upstream rename — would
     silently break MM-aware routing (workers recompute hashes internally,
     diverging from the router's pad_value substitution).
"""
from __future__ import annotations

import dataclasses
import hashlib
from types import SimpleNamespace

import pytest

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.sglang,
    pytest.mark.unit,
    pytest.mark.gpu_0,
]


def test_sglang_pad_value_formula_unchanged() -> None:
    from sglang.srt.managers import schedule_batch

    MM_PAD_SHIFT_VALUE = schedule_batch.MM_PAD_SHIFT_VALUE

    assert MM_PAD_SHIFT_VALUE == 1_000_000, (
        f"sglang MM_PAD_SHIFT_VALUE changed to {MM_PAD_SHIFT_VALUE}; "
        "dynamo's pad_value_for_sglang in lib/llm/src/preprocessor.rs "
        "must be updated in lockstep or MM-aware routing will misroute."
    )
    compute_pad_value = getattr(schedule_batch, "_compute_pad_value", None)
    if compute_pad_value is None:
        _assert_kimi_wide_padding_protocol(schedule_batch)
        return

    assert compute_pad_value(0) == 1_000_000
    fits = (1 << 30) - 1
    assert compute_pad_value(fits) == 1_000_000 + fits
    overflow = (1 << 30) | 0xCAFE
    assert compute_pad_value(overflow) == 1_000_000 + 0xCAFE, (
        "high bits above the 30-bit mask must be discarded; sglang's "
        "formula switched away from `hash % (1 << 30)` if this fails"
    )


def _assert_kimi_wide_padding_protocol(schedule_batch) -> None:
    from sglang.srt.managers.io_struct import GenerateReqInput
    from sglang.srt.multimodal.processors.kimi_k3 import KimiK3ImageProcessor

    from dynamo.sglang._compat import supports_external_mm_hashes

    item = schedule_batch.MultimodalDataItem(modality=schedule_batch.Modality.IMAGE)
    identity = bytes(range(32))
    item.set_identity(identity)
    digest = hashlib.sha256(b"sglang-mm-wide-pad-v1" + identity).digest()
    expected = tuple(
        1_000_000 + (int.from_bytes(digest[i : i + 8], "big") & ((1 << 62) - 1))
        for i in range(0, 32, 8)
    )
    assert item.pad_values == expected
    assert item.pad_value == expected[0]
    assert item.padding_sequence(6) == list(expected) + list(expected[:2])
    # Full identity matters, including bytes outside the scalar hash prefix.
    item.set_identity(identity[:8] + bytes(reversed(identity[8:])))
    assert item.pad_values != expected

    async def async_generate(mm_hashes=None):
        pass

    engine = SimpleNamespace(
        async_generate=async_generate,
        tokenizer_manager=SimpleNamespace(mm_processor=KimiK3ImageProcessor),
    )
    assert KimiK3ImageProcessor.uses_wide_image_identity is True
    assert supports_external_mm_hashes(engine) is False
    with pytest.raises(ValueError, match="Caller-supplied mm_hashes"):
        KimiK3ImageProcessor.reject_caller_image_identity(
            [], GenerateReqInput(text="test", mm_hashes=["abc"])
        )


def test_sglang_generate_req_input_has_mm_hashes_field() -> None:
    from sglang.srt.managers.io_struct import GenerateReqInput

    fields = {f.name for f in dataclasses.fields(GenerateReqInput)}
    assert "mm_hashes" in fields, (
        "GenerateReqInput.mm_hashes is missing. The dynamo sglang image "
        "was built with an older SGLang than v0.5.13, or upstream renamed "
        "the field. Without it, sglang ignores caller-supplied hashes and "
        "MM-aware KV routing silently degrades to text-prefix fallback."
    )
