# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dynamo.common.utils.engine_response."""

import pytest

from dynamo.common.utils.engine_response import normalize_finish_reason

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.core,
]


def test_abort_normalizes_to_stop():
    # Regression test for the crash observed in production logs:
    # "unknown variant `abort`, expected one of `stop`, `length`,
    # `tool_calls`, `content_filter`, `function_call`" -- the external
    # dynamo-protocols FinishReason enum has no Cancelled/Abort variant,
    # so the normalized value MUST be one of its 5 valid variants.
    assert normalize_finish_reason("abort") == "stop"


def test_abort_with_suffix_normalizes_to_stop():
    # startswith("abort") guard must remain permissive for any
    # "abort: <detail>"-style value the engine might emit.
    assert normalize_finish_reason("abort: preempted") == "stop"


def test_stop_passes_through_unchanged():
    assert normalize_finish_reason("stop") == "stop"


def test_length_passes_through_unchanged():
    assert normalize_finish_reason("length") == "length"


def test_none_passes_through_unchanged():
    assert normalize_finish_reason(None) is None


def test_empty_string_passes_through_unchanged():
    assert normalize_finish_reason("") == ""


def test_normalized_value_is_wire_compatible():
    # Explicit assertion of the actual invariant this fix protects:
    # the output must be a member of the external wire schema's 5
    # valid finish_reason values (stop/length/tool_calls/
    # content_filter/function_call) -- never "cancelled" or "abort".
    wire_compatible_values = {
        "stop",
        "length",
        "tool_calls",
        "content_filter",
        "function_call",
    }
    assert normalize_finish_reason("abort") in wire_compatible_values
