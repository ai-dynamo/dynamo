# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for output_formatter.py — modality-specific formatters."""

from unittest.mock import MagicMock

import pytest

from dynamo.vllm.omni.output_formatter import (
    TextFormatter,
    _build_completion_usage,
    _error_chunk,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_1,
    pytest.mark.pre_merge,
]


# ── TextFormatter ──────────────────────────────────────────


def _make_request_output(text="hello world", finish_reason=None):
    output = MagicMock()
    output.text = text
    output.finish_reason = finish_reason
    output.token_ids = [1, 2, 3]  # 3 completion tokens
    ro = MagicMock()
    ro.outputs = [output]
    ro.prompt_token_ids = [
        10,
        20,
        30,
        40,
        50,
    ]  # 5 prompt tokens (different from completion)
    return ro


class TestTextFormatter:
    def test_delta_text(self):
        f = TextFormatter(model_name="test-model")
        chunk = f.format(
            _make_request_output("hello world"), "req-1", previous_text="hello "
        )
        assert chunk["choices"][0]["delta"]["content"] == "world"

    def test_no_outputs_returns_error(self):
        f = TextFormatter(model_name="test-model")
        ro = MagicMock()
        ro.outputs = []
        chunk = f.format(ro, "req-1")
        assert "Error" in chunk["choices"][0]["delta"]["content"]

    def test_finish_reason_included(self):
        f = TextFormatter(model_name="test-model")
        ro = _make_request_output("done", finish_reason="stop")
        chunk = f.format(ro, "req-1")
        assert chunk["choices"][0]["finish_reason"] == "stop"
        assert "usage" in chunk

    def test_finish_reason_abort_normalized(self):
        f = TextFormatter(model_name="test-model")
        ro = _make_request_output("done", finish_reason="abort")
        chunk = f.format(ro, "req-1")
        assert chunk["choices"][0]["finish_reason"] == "cancelled"

    def test_finish_reason_none_when_not_finished(self):
        f = TextFormatter(model_name="test-model")
        ro = _make_request_output("partial")
        chunk = f.format(ro, "req-1")
        assert chunk["choices"][0]["finish_reason"] is None

    def test_model_name_in_response(self):
        f = TextFormatter(model_name="my-model")
        chunk = f.format(_make_request_output(), "req-1")
        assert chunk["model"] == "my-model"

    def test_usage_has_prompt_and_completion_tokens(self):
        f = TextFormatter(model_name="test-model")
        ro = _make_request_output("done", finish_reason="stop")
        chunk = f.format(ro, "req-1")
        assert chunk["usage"]["prompt_tokens"] == 5  # 5 prompt token IDs
        assert chunk["usage"]["completion_tokens"] == 3  # 3 completion token IDs
        assert chunk["usage"]["total_tokens"] == 8


# ── Helpers ────────────────────────────────────────────────


class TestErrorChunk:
    def test_error_chunk_format(self):
        chunk = _error_chunk("req-1", "my-model", "something broke")
        assert chunk["choices"][0]["delta"]["content"] == "Error: something broke"
        assert chunk["choices"][0]["finish_reason"] == "error"
        assert chunk["model"] == "my-model"


class TestBuildCompletionUsage:
    def test_basic(self):
        ro = _make_request_output("hello", finish_reason="stop")
        usage = _build_completion_usage(ro)
        assert usage["prompt_tokens"] == 5
        assert usage["completion_tokens"] == 3
        assert usage["total_tokens"] == 8

    def test_no_prompt_tokens(self):
        ro = _make_request_output()
        ro.prompt_token_ids = None
        usage = _build_completion_usage(ro)
        assert usage["prompt_tokens"] is None
        assert usage["total_tokens"] is None
