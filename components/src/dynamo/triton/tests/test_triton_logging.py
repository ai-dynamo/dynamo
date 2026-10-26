# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for create_triton_log_callback(): the adapter that forwards Triton
server log records into the Dynamo worker's Python logging pipeline."""

import logging

import pytest
from tritonserver._c.triton_bindings import TRITONSERVER_LogLevel as LogLevel

from dynamo.triton.util import create_triton_log_callback

pytestmark = [
    pytest.mark.unit,
    pytest.mark.triton,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.fixture
def make_log_forwarder(monkeypatch, request):
    """Build a create_triton_log_callback() whose logger.handle() is captured.

    Returns make(level=DEBUG) -> (forward, records); `records` collects the
    LogRecords the forwarder emits. We patch handle() to capture the LogRecords
    so we can assert their contents.
    """

    def _make(level: int = logging.DEBUG, logger_name: str = "triton_unit_test"):
        log = logging.getLogger(logger_name)
        # restore the level after the test
        original_level = log.level
        request.addfinalizer(lambda: log.setLevel(original_level))
        log.setLevel(level)
        records: list[logging.LogRecord] = []
        monkeypatch.setattr(log, "handle", records.append)
        return create_triton_log_callback(logger_name), records

    return _make


@pytest.mark.parametrize(
    "triton_level, expected_level",
    [
        (LogLevel.INFO, logging.INFO),
        (LogLevel.WARN, logging.WARNING),
        (LogLevel.ERROR, logging.ERROR),
        (LogLevel.VERBOSE, logging.DEBUG),
    ],
    ids=["INFO", "WARN", "ERROR", "VERBOSE"],
)
def test_log_callback_maps_level_and_forwards_record(
    make_log_forwarder, triton_level, expected_level
):
    """For every real Triton level, the record gets the expected Python level and
    carries Triton's message, source file, line, and func unchanged."""
    forward, records = make_log_forwarder()
    forward(
        triton_level,
        "cache_manager.cc",
        480,
        1_782_000_000_000,
        "Create CacheManager with cache_dir: '/opt/tritonserver/caches'",
    )

    (record,) = records
    assert record.levelno == expected_level
    assert record.getMessage() == (
        "Create CacheManager with cache_dir: '/opt/tritonserver/caches'"
    )
    assert record.pathname == "cache_manager.cc"
    assert record.lineno == 480
    assert record.funcName == "<module>"


def test_log_callback_drops_verbose_below_debug(make_log_forwarder):
    """Drops verbose (DEBUG) log records unless logger level is set to DEBUG."""
    forward, records = make_log_forwarder(level=logging.INFO)

    forward(
        LogLevel.VERBOSE,
        "model_lifecycle.cc",
        340,
        0,
        "GetModel() 'add_sub' version -1",
    )
    assert records == []  # verbose suppressed while the logger is at INFO

    forward(LogLevel.INFO, "model_lifecycle.cc", 473, 0, "loading: add_sub:1")
    assert len(records) == 1
    assert records[0].getMessage() == "loading: add_sub:1"
