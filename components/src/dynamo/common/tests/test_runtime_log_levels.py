# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the level names LogHandler passes to the Rust log bridge."""

import logging

import pytest

import dynamo.runtime.logging as dynamo_logging

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.mark.parametrize(
    ("level", "expected"),
    [
        (logging.CRITICAL, "error"),
        (35, "warn"),
        (15, "debug"),
        (5, "debug"),
    ],
)
def test_handler_sends_level_rust_knows(monkeypatch, level, expected):
    sent = []
    monkeypatch.setattr(
        dynamo_logging,
        "log_message",
        lambda level, message, module, file, line: sent.append(level),
    )
    record = logging.LogRecord("test", level, __file__, 1, "message", None, None)

    dynamo_logging.LogHandler().handle(record)

    assert sent == [expected]
