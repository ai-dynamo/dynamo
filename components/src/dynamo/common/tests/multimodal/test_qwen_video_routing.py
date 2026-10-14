# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the shared Qwen video resize probe."""

import logging
from unittest.mock import Mock

import pytest

from dynamo.common.multimodal.qwen_video_routing import (
    QWEN_VIDEO_RESIZE_LEGACY_CEIL,
    QWEN_VIDEO_RESIZE_ROUND_TIES_EVEN,
    resolve_qwen_video_resize_mode,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.multimodal,
]

LOGGER_NAME = "dynamo.test.qwen_video_routing"
VLLM_PREFIX = "Exact video-aware KV routing disabled"
SGLANG_PREFIX = "Exact SGLang video-aware KV routing disabled"

EXPECTED_PROBE_KWARGS = {
    "num_frames": 5,
    "height": 1120,
    "width": 3760,
    "temporal_factor": 2,
    "factor": 32,
    "min_pixels": 4096,
    "max_pixels": 25165824,
}


def _resolve(smart_resize, prefix=VLLM_PREFIX):
    return resolve_qwen_video_resize_mode(
        smart_resize,
        logger=logging.getLogger(LOGGER_NAME),
        disabled_prefix=prefix,
    )


def _warnings(caplog):
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == LOGGER_NAME and record.levelno == logging.WARNING
    ]


def test_resize_mode_constants():
    assert QWEN_VIDEO_RESIZE_LEGACY_CEIL == "legacy_ceil"
    assert QWEN_VIDEO_RESIZE_ROUND_TIES_EVEN == "round_ties_even"


def test_probe_calls_smart_resize_once_with_exact_arguments():
    smart_resize = Mock(return_value=(1216, 4096))

    _resolve(smart_resize)

    smart_resize.assert_called_once_with(**EXPECTED_PROBE_KWARGS)


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ((1216, 4096), "legacy_ceil"),
        ((1120, 3776), "round_ties_even"),
    ],
)
def test_recognized_results_map_to_resize_mode(caplog, result, expected):
    caplog.set_level(logging.WARNING, logger=LOGGER_NAME)

    assert _resolve(Mock(return_value=result)) == expected
    assert _warnings(caplog) == []


@pytest.mark.parametrize(
    ("prefix", "expected"),
    [
        (
            VLLM_PREFIX,
            "Exact video-aware KV routing disabled because the installed Qwen "
            "smart_resize behavior is unsupported: (999, 999)",
        ),
        (
            SGLANG_PREFIX,
            "Exact SGLang video-aware KV routing disabled because the installed "
            "Qwen smart_resize behavior is unsupported: (999, 999)",
        ),
    ],
)
def test_unknown_result_returns_none_and_warns(caplog, prefix, expected):
    caplog.set_level(logging.WARNING, logger=LOGGER_NAME)

    assert _resolve(Mock(return_value=(999, 999)), prefix) is None
    assert _warnings(caplog) == [expected]


@pytest.mark.parametrize(
    ("prefix", "expected"),
    [
        (
            VLLM_PREFIX,
            "Exact video-aware KV routing disabled because the installed "
            "Transformers package has no Qwen3 video processor",
        ),
        (
            SGLANG_PREFIX,
            "Exact SGLang video-aware KV routing disabled because the installed "
            "Transformers package has no Qwen3 video processor",
        ),
    ],
)
def test_absent_smart_resize_returns_none_and_warns(caplog, prefix, expected):
    caplog.set_level(logging.WARNING, logger=LOGGER_NAME)

    assert _resolve(None, prefix) is None
    assert _warnings(caplog) == [expected]


@pytest.mark.parametrize("error_type", [TypeError, ValueError])
@pytest.mark.parametrize(
    ("prefix", "expected_start"),
    [
        (
            VLLM_PREFIX,
            "Exact video-aware KV routing disabled because the installed Qwen "
            "smart_resize API is unsupported: ",
        ),
        (
            SGLANG_PREFIX,
            "Exact SGLang video-aware KV routing disabled because the installed "
            "Qwen smart_resize API is unsupported: ",
        ),
    ],
)
def test_unsupported_api_returns_none_and_warns(
    caplog, error_type, prefix, expected_start
):
    caplog.set_level(logging.WARNING, logger=LOGGER_NAME)
    smart_resize = Mock(side_effect=error_type("bad probe"))

    assert _resolve(smart_resize, prefix) is None
    assert _warnings(caplog) == [expected_start + "bad probe"]


def test_other_exceptions_propagate():
    smart_resize = Mock(side_effect=RuntimeError("unexpected"))

    with pytest.raises(RuntimeError, match="unexpected"):
        _resolve(smart_resize)
