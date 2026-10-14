# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any

import pytest

from dynamo.common.multimodal.qwen_video_contract import (
    QWEN_VIDEO_RESIZE_LEGACY_CEIL,
    QWEN_VIDEO_RESIZE_ROUND_TIES_EVEN,
    resolve_qwen_video_resize_mode,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]

LOGGER_NAME = "dynamo.test.qwen_video_contract"
VLLM_FEATURE = "video-aware KV routing"
SGLANG_FEATURE = "SGLang video-aware KV routing"


def _resolve(resize: Any, feature: str = VLLM_FEATURE) -> Any:
    return resolve_qwen_video_resize_mode(
        resize, logger=logging.getLogger(LOGGER_NAME), feature=feature
    )


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == LOGGER_NAME and record.levelno == logging.WARNING
    ]


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ((1216, 4096), QWEN_VIDEO_RESIZE_LEGACY_CEIL),
        ((1120, 3776), QWEN_VIDEO_RESIZE_ROUND_TIES_EVEN),
    ],
)
def test_recognized_resize_results(
    caplog: pytest.LogCaptureFixture, result: tuple[int, int], expected: str
) -> None:
    calls: list[dict[str, Any]] = []

    def resize(**kwargs: Any) -> tuple[int, int]:
        calls.append(kwargs)
        return result

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        assert _resolve(resize) == expected
    assert calls == [
        {
            "num_frames": 5,
            "height": 1120,
            "width": 3760,
            "temporal_factor": 2,
            "factor": 32,
            "min_pixels": 4096,
            "max_pixels": 25165824,
        }
    ]
    assert _warnings(caplog) == []


@pytest.mark.parametrize(
    ("feature", "subject"),
    [
        (VLLM_FEATURE, "Exact video-aware KV routing"),
        (SGLANG_FEATURE, "Exact SGLang video-aware KV routing"),
    ],
)
def test_missing_resize_callable(
    caplog: pytest.LogCaptureFixture, feature: str, subject: str
) -> None:
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        assert _resolve(None, feature) is None
    assert _warnings(caplog) == [
        f"{subject} disabled because the installed "
        "Transformers package has no Qwen3 video processor"
    ]


@pytest.mark.parametrize(
    ("feature", "subject"),
    [
        (VLLM_FEATURE, "Exact video-aware KV routing"),
        (SGLANG_FEATURE, "Exact SGLang video-aware KV routing"),
    ],
)
def test_unexpected_resize_result(
    caplog: pytest.LogCaptureFixture, feature: str, subject: str
) -> None:
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        assert _resolve(lambda **_: (1, 2), feature) is None
    assert _warnings(caplog) == [
        f"{subject} disabled because the installed Qwen "
        "smart_resize behavior is unsupported: (1, 2)"
    ]


@pytest.mark.parametrize("error_type", [TypeError, ValueError])
@pytest.mark.parametrize(
    ("feature", "subject"),
    [
        (VLLM_FEATURE, "Exact video-aware KV routing"),
        (SGLANG_FEATURE, "Exact SGLang video-aware KV routing"),
    ],
)
def test_unsupported_resize_api(
    caplog: pytest.LogCaptureFixture,
    error_type: type[Exception],
    feature: str,
    subject: str,
) -> None:
    def resize(**_: Any) -> tuple[int, int]:
        raise error_type("bad probe")

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        assert _resolve(resize, feature) is None
    assert _warnings(caplog) == [
        f"{subject} disabled because the installed Qwen "
        "smart_resize API is unsupported: bad probe"
    ]


def test_other_errors_propagate() -> None:
    def resize(**_: Any) -> tuple[int, int]:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        _resolve(resize)
