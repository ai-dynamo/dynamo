# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pin sglang's MM-routing wire-protocol contracts against drift.

These fast unit tests import the installed sglang module and verify two
contracts that the dynamo Rust frontend depends on for MM-aware KV
routing:

  1. After SGLang applies a caller hash and Dynamo's compatibility repair
     runs, every media position must hold
     ``MM_PAD_SHIFT_VALUE + (hash % (1 << 30))`` with
     ``MM_PAD_SHIFT_VALUE == 1_000_000``. Mirrored in Rust at
     ``dynamo_kv_router::protocols::pad_value_for_mm_hash`` (shared by the
     frontend and the kv-router's vLLM-event normalization). Pinned by
     the Rust unit ``pad_value_matches_sglang_protocol`` on our side;
     this test pins the same constants on sglang's side so a future
     sglang bump that shifts the formula fails-closed at PR time.

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


def test_sglang_pad_value_formula_unchanged() -> None:
    import asyncio
    from types import SimpleNamespace

    from sglang.srt.managers.schedule_batch import (
        MM_PAD_SHIFT_VALUE,
        Modality,
        MultimodalDataItem,
    )

    from dynamo.common.multimodal.routing_utils import pad_value_for_mm_hash
    from dynamo.sglang._compat import supports_external_mm_hashes

    assert MM_PAD_SHIFT_VALUE == 1_000_000, (
        f"sglang MM_PAD_SHIFT_VALUE changed to {MM_PAD_SHIFT_VALUE}; "
        "dynamo's pad_value_for_mm_hash routing protocol "
        "must be updated in lockstep or MM-aware routing will misroute."
    )
    assert pad_value_for_mm_hash(0) == 1_000_000
    fits = (1 << 30) - 1
    assert pad_value_for_mm_hash(fits) == 1_000_000 + fits
    assert pad_value_for_mm_hash((1 << 30) | 0xCAFE) == 1_000_000 + 0xCAFE

    async def tokenize(obj):
        item = MultimodalDataItem(modality=Modality.IMAGE, offsets=[(1, 5)])
        item.set_hash(int(obj.mm_hashes[0], 16))
        mm_inputs = SimpleNamespace(mm_items=[item], padded_input_ids=None)
        return SimpleNamespace(input_ids=[7] * 7, mm_inputs=mm_inputs)

    class Engine:
        tokenizer_manager = SimpleNamespace(_tokenize_one_request=tokenize)

        async def async_generate(self, prompt=None, mm_hashes=None):
            return None

    engine = Engine()
    assert supports_external_mm_hashes(engine)
    tokenize_one = engine.tokenizer_manager._tokenize_one_request
    for mm_hash in (0, (1 << 30) - 1, (1 << 30) | 0xCAFE, 0x9F3A6C1D2E4B5A78):
        request = SimpleNamespace(mm_hashes=[f"{mm_hash:016x}"])
        mm_inputs = asyncio.run(tokenize_one(request)).mm_inputs
        expected = [pad_value_for_mm_hash(mm_hash)] * 5
        assert mm_inputs.padded_input_ids[1:6] == expected, (
            f"mm_hash {mm_hash:#x} is padded as {mm_inputs.padded_input_ids[1:6]}, "
            f"but the KV router assumes {expected}."
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
