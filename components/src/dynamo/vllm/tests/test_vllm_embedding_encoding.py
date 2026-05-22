# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the vLLM embedding handler's base64 wire-format helper.

The handler-level integration with ``encoding_format`` is covered by
``tests/serve/test_vllm.py::embedding_agg`` (which speaks to a live
engine); this file pins the helper's byte-exact behavior so a refactor
that breaks compatibility with the SGLang side -- or with OpenAI
clients -- fails fast in unit CI.
"""

import base64
import struct

import pytest

from dynamo.vllm.handlers import _encode_floats_to_base64

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _decode_base64_floats(encoded: str, expected_dim: int) -> list[float]:
    raw = base64.b64decode(encoded)
    assert len(raw) == expected_dim * 4
    return list(struct.unpack(f"<{expected_dim}f", raw))


def test_encode_helper_matches_struct_pack():
    """Byte-exact match with little-endian f32 packing -- this is the
    OpenAI wire format and what every official SDK assumes."""
    floats = [0.0, 1.0, -1.0, 3.14]
    encoded = _encode_floats_to_base64(floats)
    expected = base64.b64encode(struct.pack("<4f", *floats)).decode("ascii")
    assert encoded == expected
    assert _decode_base64_floats(encoded, 4) == pytest.approx(floats)


def test_encode_helper_matches_sglang_side():
    """vLLM and SGLang must emit the same bytes for the same float
    vector. Drift here would break clients that load-balance across
    backends; the assertion is a cheap canary.

    Skipped automatically when the sglang package isn't importable in
    this CI image -- the vllm-runtime container ships vLLM only.
    """
    sglang_handler = pytest.importorskip(
        "dynamo.sglang.request_handlers.embedding.embedding_handler",
        reason="sglang not installed in this runtime image",
    )
    floats = [0.123, -0.456, 0.789, 1.0, -1.0]
    assert _encode_floats_to_base64(floats) == sglang_handler._encode_floats_to_base64(
        floats
    )


def test_encode_helper_handles_empty_vector():
    assert _encode_floats_to_base64([]) == ""


def test_encode_helper_handles_large_vector():
    """Smoke test for typical embedding dimensionalities (1024+). The
    output length should be ``ceil(dim * 4 / 3) * 4`` -- standard base64
    padding rules."""
    floats = [float(i) * 0.001 for i in range(1024)]
    encoded = _encode_floats_to_base64(floats)
    decoded = _decode_base64_floats(encoded, 1024)
    assert decoded == pytest.approx(floats)
