# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the SGLang embedding handler's ``encoding_format`` path.

The default ``"float"`` path is exercised end-to-end by
``test_sglang_embedding_metrics.py``; this file focuses on the
``"base64"`` path: the wire-format helper and the handler's
``_transform_response`` integration with it.
"""

import base64
import struct

import pytest

from dynamo.sglang.request_handlers.embedding import embedding_handler as eh

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
]


def _decode_base64_floats(encoded: str, expected_dim: int) -> list[float]:
    """Reverse the OpenAI base64 wire format back to a float vector.

    Mirrors what an OpenAI-compatible client (e.g. the official Python
    SDK) does after receiving an ``encoding_format=base64`` response:
    base64-decode, then interpret the bytes as little-endian ``f32``.
    """
    raw = base64.b64decode(encoded)
    assert len(raw) == expected_dim * 4, (
        f"expected {expected_dim * 4} bytes for {expected_dim}-dim f32, "
        f"got {len(raw)}"
    )
    return list(struct.unpack(f"<{expected_dim}f", raw))


def test_encode_helper_matches_struct_pack():
    """The helper produces standard-alphabet base64 of little-endian f32 bytes."""
    floats = [0.0, 1.0, -1.0, 3.14]
    encoded = eh._encode_floats_to_base64(floats)
    expected = base64.b64encode(struct.pack("<4f", *floats)).decode("ascii")
    assert encoded == expected
    # Sanity round-trip.
    assert _decode_base64_floats(encoded, 4) == pytest.approx(floats)


def test_encode_helper_handles_empty_vector():
    """A zero-dimension vector produces the empty base64 string."""
    assert eh._encode_floats_to_base64([]) == ""


def test_transform_response_base64_wraps_each_embedding(monkeypatch):
    """``_transform_response`` returns base64 strings (not lists) for every
    per-input embedding when ``encoding_format="base64"`` is set, while
    preserving the surrounding response shape."""
    monkeypatch.setattr(
        "dynamo.sglang.request_handlers.handler_base.BaseWorkerHandler.__init__",
        lambda self, *a, **kw: None,
    )
    # Silence the metrics path -- this test is about wire shape only.
    monkeypatch.setattr(eh, "observe_embedding_batch_size", lambda *a, **kw: None)
    monkeypatch.setattr(eh, "observe_embedding_input_tokens", lambda *a, **kw: None)

    handler = eh.EmbeddingWorkerHandler.__new__(eh.EmbeddingWorkerHandler)

    ret = [
        {"embedding": [0.1, 0.2, 0.3], "meta_info": {"prompt_tokens": 12}},
        {"embedding": [0.4, 0.5, 0.6], "meta_info": {"prompt_tokens": 8}},
    ]
    out = handler._transform_response(
        ret, "Qwen/Qwen3-Embedding-4B", encoding_format="base64"
    )

    assert out["object"] == "list"
    assert len(out["data"]) == 2
    for idx, expected_floats in enumerate([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]):
        payload = out["data"][idx]["embedding"]
        assert isinstance(payload, str), (
            f"embedding {idx} should be a base64 string under base64 encoding, "
            f"got {type(payload).__name__}"
        )
        decoded = _decode_base64_floats(payload, len(expected_floats))
        assert decoded == pytest.approx(expected_floats)


def test_transform_response_base64_applies_dimensions_truncation_first(monkeypatch):
    """``dimensions`` truncation must run before base64 encoding so the
    byte count matches the requested dimensionality (a client decoding
    with ``struct.unpack('<{dim}f', ...)`` will otherwise miscount)."""
    monkeypatch.setattr(
        "dynamo.sglang.request_handlers.handler_base.BaseWorkerHandler.__init__",
        lambda self, *a, **kw: None,
    )
    monkeypatch.setattr(eh, "observe_embedding_batch_size", lambda *a, **kw: None)
    monkeypatch.setattr(eh, "observe_embedding_input_tokens", lambda *a, **kw: None)

    handler = eh.EmbeddingWorkerHandler.__new__(eh.EmbeddingWorkerHandler)

    full = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    ret = [{"embedding": full, "meta_info": {"prompt_tokens": 1}}]
    out = handler._transform_response(
        ret, "model-A", dimensions=3, encoding_format="base64"
    )

    payload = out["data"][0]["embedding"]
    assert isinstance(payload, str)
    decoded = _decode_base64_floats(payload, 3)
    assert decoded == pytest.approx(full[:3])


def test_transform_response_defaults_to_float(monkeypatch):
    """Default behavior (no ``encoding_format`` argument) returns float lists
    -- preserves the wire shape for clients that have not opted into base64."""
    monkeypatch.setattr(
        "dynamo.sglang.request_handlers.handler_base.BaseWorkerHandler.__init__",
        lambda self, *a, **kw: None,
    )
    monkeypatch.setattr(eh, "observe_embedding_batch_size", lambda *a, **kw: None)
    monkeypatch.setattr(eh, "observe_embedding_input_tokens", lambda *a, **kw: None)

    handler = eh.EmbeddingWorkerHandler.__new__(eh.EmbeddingWorkerHandler)
    ret = [{"embedding": [0.1, 0.2, 0.3], "meta_info": {"prompt_tokens": 1}}]
    out = handler._transform_response(ret, "m")
    assert out["data"][0]["embedding"] == [0.1, 0.2, 0.3]
