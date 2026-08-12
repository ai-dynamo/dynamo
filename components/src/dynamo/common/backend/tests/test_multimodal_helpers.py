# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the unified-backend multimodal contract helpers.

Pin the wire shape: extract_multimodal_kwargs omits absent/None-valued
keys (but passes through empty dicts), encoder_terminal_chunk produces
the exact terminal-chunk shape Rust producers emit, and
require_encoder_result rejects non-object payloads.
"""

from __future__ import annotations

import pytest

from dynamo.common.backend.multimodal import (
    encoder_terminal_chunk,
    extract_multimodal_kwargs,
    require_encoder_result,
)
from dynamo.common.constants import DisaggregationMode

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]

# ---------------------------------------------------------------------------
# extract_multimodal_kwargs
# ---------------------------------------------------------------------------


def test_extract_multimodal_kwargs_round_trips_image_video_audio():
    """Image/video/audio URL maps survive unchanged through the helper.
    Engines splat the result onto APIs that accept multi_modal_data /
    mm_processor_kwargs directly, so the helper must not mutate."""
    request = {
        "token_ids": [1, 2, 3],
        "multi_modal_data": {
            "image": [{"Url": "https://example.com/cat.png"}],
            "video": [{"Url": "https://example.com/clip.mp4"}],
            "audio": [{"Url": "https://example.com/voice.wav"}],
        },
        "mm_processor_kwargs": {"use_audio_in_video": True},
    }
    kwargs = extract_multimodal_kwargs(request)
    assert kwargs == {
        "multi_modal_data": request["multi_modal_data"],
        "mm_processor_kwargs": {"use_audio_in_video": True},
    }


def test_extract_multimodal_kwargs_returns_none_for_text_only_request():
    """Non-MM requests should return None so callers can use
    `extract_multimodal_kwargs(req) or {}` without a branch."""
    request = {"token_ids": [1, 2, 3]}
    assert extract_multimodal_kwargs(request) is None


def test_extract_multimodal_kwargs_omits_absent_keys():
    """Only `multi_modal_data` present -> result has only that key, not
    `mm_processor_kwargs: None`."""
    request = {
        "token_ids": [1],
        "multi_modal_data": {"image": [{"Url": "https://example.com/x.png"}]},
    }
    kwargs = extract_multimodal_kwargs(request)
    assert kwargs == {"multi_modal_data": request["multi_modal_data"]}
    assert "mm_processor_kwargs" not in kwargs


def test_extract_multimodal_kwargs_omits_none_valued_keys():
    """`mm_processor_kwargs: None` is treated as absent. Engines that
    interpret `None` differently from a missing kwarg must not see the
    None value leak through."""
    request = {
        "token_ids": [1],
        "multi_modal_data": {"image": [{"Url": "x"}]},
        "mm_processor_kwargs": None,
    }
    kwargs = extract_multimodal_kwargs(request)
    assert kwargs == {"multi_modal_data": request["multi_modal_data"]}
    assert "mm_processor_kwargs" not in kwargs


def test_extract_multimodal_kwargs_passes_through_empty_dict():
    """An empty dict is a legitimate value (framework guarantees shape,
    not content). Callers see `multi_modal_data: {}` unchanged."""
    request = {
        "token_ids": [1],
        "multi_modal_data": {},
    }
    kwargs = extract_multimodal_kwargs(request)
    assert kwargs == {"multi_modal_data": {}}


# ---------------------------------------------------------------------------
# require_encoder_result
# ---------------------------------------------------------------------------


def test_require_encoder_result_returns_dict_as_is():
    """The dict the encoder emitted is the dict the consumer receives --
    no unwrap of nested keys."""
    payload = {
        "embedding_handle": {
            "uri": "nixl://encoder-0/embedding-42",
            "shape": [1, 1024],
            "dtype": "fp16",
        },
    }
    request = {"token_ids": [1], "encoder_result": payload}
    assert require_encoder_result(request, DisaggregationMode.PREFILL) == payload


def test_require_encoder_result_raises_with_mode_name_when_missing():
    """The error message must name the role (using ``mode.value``, like
    the sibling ``require_prefill_result``) so operators can tell which
    worker received the malformed request."""
    request = {"token_ids": [1]}
    with pytest.raises(ValueError, match="agg worker received request"):
        require_encoder_result(request, DisaggregationMode.AGGREGATED)
    with pytest.raises(ValueError, match="prefill worker received request"):
        require_encoder_result(request, DisaggregationMode.PREFILL)


@pytest.mark.parametrize(
    "bad_payload",
    [
        [1, 2, 3],
        "string",
        42,
        True,
    ],
)
def test_require_encoder_result_rejects_non_object_payload(bad_payload):
    """Non-object payloads break the `.get(key)` consumer contract --
    reject loudly instead of letting an array or scalar reach engines."""
    request = {"token_ids": [1], "encoder_result": bad_payload}
    with pytest.raises(ValueError, match="must be a JSON object"):
        require_encoder_result(request, DisaggregationMode.PREFILL)


# ---------------------------------------------------------------------------
# encoder_terminal_chunk
# ---------------------------------------------------------------------------


def test_encoder_terminal_chunk_exact_shape():
    """Pinned shape: empty token_ids, index=0, finish_reason='stop',
    encoder_result carried through unchanged. Matches the Rust
    LLMEngineOutput::encode_terminal constructor byte-for-byte."""
    payload = {"embedding_handle": {"uri": "nixl://encoder/0"}}
    chunk = encoder_terminal_chunk(payload)
    assert chunk == {
        "token_ids": [],
        "index": 0,
        "finish_reason": "stop",
        "encoder_result": payload,
    }


def test_encoder_terminal_chunk_omits_completion_usage_when_none():
    """completion_usage absent when caller doesn't pass it -- matches
    the TypedDict total=False semantics and the Rust skip_serializing_if
    style."""
    chunk = encoder_terminal_chunk({"x": 1})
    assert "completion_usage" not in chunk


def test_encoder_terminal_chunk_includes_completion_usage_when_given():
    """When caller passes a usage dict, the chunk carries it."""
    usage = {"prompt_tokens": 32, "completion_tokens": 0, "total_tokens": 32}
    chunk = encoder_terminal_chunk({"x": 1}, completion_usage=usage)
    assert chunk["completion_usage"] == usage


def test_encoder_terminal_chunk_accepts_empty_dict():
    """Empty dict is a valid encoder_result. Framework guarantees the
    shape (object), engines own content validation."""
    chunk = encoder_terminal_chunk({})
    assert chunk["encoder_result"] == {}


@pytest.mark.parametrize(
    "bad_payload",
    [
        [1, 2, 3],
        "string",
        42,
        None,
    ],
)
def test_encoder_terminal_chunk_rejects_non_dict_payload(bad_payload):
    """Non-dict input is rejected at construction so engine bugs surface
    where they originate, not at the downstream consumer."""
    with pytest.raises(TypeError, match="encoder_result must be a dict"):
        encoder_terminal_chunk(bad_payload)


# ---------------------------------------------------------------------------
# End-to-end wire shape: producer -> consumer is no-wrapper
# ---------------------------------------------------------------------------


def test_encoder_result_no_wrapper_end_to_end():
    """The dict an Encode worker emits via encoder_terminal_chunk is the
    same dict a downstream worker reads via require_encoder_result. No
    nested 'encoder_payload' key, no wrapper struct -- single shape end
    to end."""
    original = {
        "embedding_handle": {"uri": "nixl://e/0", "shape": [1, 1024]},
        "processed_token_ids": [128_000, 200_001, 200_002],
    }
    terminal = encoder_terminal_chunk(original)
    # Forwarded by the frontend onto the downstream request:
    downstream_request = {
        "token_ids": [1],
        "encoder_result": terminal["encoder_result"],
    }
    consumed = require_encoder_result(downstream_request, DisaggregationMode.PREFILL)
    assert consumed == original
    assert consumed is terminal["encoder_result"]
