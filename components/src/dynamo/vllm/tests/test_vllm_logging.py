# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Regression tests for vLLM logging integration.

vLLM configures its own logger at import time (before dynamo can act).
These tests verify that dynamo reclaims the vllm logger in the main process,
that VLLM_LOGGING_LEVEL controls the vllm logger level (not DYN_LOG),
and that no config file is written (subprocesses use vLLM's built-in default).
"""

import logging
import os

import pytest

from dynamo.runtime.logging import (
    VllmColorFormatter,
    configure_dynamo_logging,
    configure_vllm_logging,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.pre_merge,
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Remove logging-related env vars before each test."""
    for var in [
        "DYN_LOG",
        "VLLM_LOGGING_LEVEL",
        "VLLM_CONFIGURE_LOGGING",
        "VLLM_LOGGING_CONFIG_PATH",
        "DYN_SKIP_SGLANG_LOG_FORMATTING",
        "SGLANG_LOGGING_CONFIG_PATH",
    ]:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def _clean_loggers():
    """Reset the root and vllm loggers before/after each test."""

    def _reset():
        for name in (None, "vllm"):
            lgr = logging.getLogger(name)
            lgr.handlers.clear()
            lgr.setLevel(logging.WARNING if name is None else logging.NOTSET)
        logging.getLogger("vllm").propagate = True
        logging.getLogger("vllm.v1.engine.async_llm").filters.clear()

    _reset()
    yield
    _reset()


def _simulate_vllm_import_time_logger():
    """Reproduce what vLLM's _configure_vllm_root_logger() does at import time."""
    vllm_logger = logging.getLogger("vllm")
    vllm_logger.addHandler(logging.StreamHandler())
    vllm_logger.propagate = False
    return vllm_logger


def test_dictconfig_replaces_vllm_stream_handler():
    """
    Regression: vLLM sets up a StreamHandler at import time. configure_vllm_logging()
    must replace it with a new StreamHandler using VllmColorFormatter so logs
    bypass the Rust bridge and respect VLLM_LOGGING_LEVEL independently.
    """
    vllm_logger = _simulate_vllm_import_time_logger()

    configure_vllm_logging(logging.INFO)

    assert len(vllm_logger.handlers) == 1
    assert isinstance(vllm_logger.handlers[0], logging.StreamHandler)
    assert isinstance(vllm_logger.handlers[0].formatter, VllmColorFormatter)


def test_vllm_logging_level_controls_vllm_logger(monkeypatch):
    """
    VLLM_LOGGING_LEVEL=DEBUG must set the vllm logger to DEBUG,
    regardless of DYN_LOG.
    """
    monkeypatch.setenv("VLLM_LOGGING_LEVEL", "DEBUG")

    configure_vllm_logging(logging.INFO)

    vllm_logger = logging.getLogger("vllm")
    assert vllm_logger.level == logging.DEBUG
    assert isinstance(vllm_logger.handlers[0], logging.StreamHandler)
    assert isinstance(vllm_logger.handlers[0].formatter, VllmColorFormatter)


def test_dyn_log_does_not_affect_vllm_level(monkeypatch):
    """
    DYN_LOG=debug alone must NOT change the vllm logger level from the
    default INFO. DYN_LOG controls dynamo logging only.
    """
    monkeypatch.setenv("DYN_LOG", "debug")

    configure_dynamo_logging()

    vllm_logger = logging.getLogger("vllm")
    assert vllm_logger.level == logging.INFO
    assert isinstance(vllm_logger.handlers[0], logging.StreamHandler)
    assert isinstance(vllm_logger.handlers[0].formatter, VllmColorFormatter)


def test_no_config_file_written():
    """
    configure_vllm_logging() must NOT set VLLM_LOGGING_CONFIG_PATH.
    Subprocesses should use vLLM's built-in DEFAULT_LOGGING_CONFIG instead
    of a config file pointing to dynamo's LogHandler (which requires the
    Rust runtime that is not available in subprocesses).
    """
    configure_vllm_logging(logging.INFO)

    assert "VLLM_LOGGING_CONFIG_PATH" not in os.environ
    assert os.environ.get("VLLM_CONFIGURE_LOGGING") == "1"


def test_health_check_filter_applied():
    """
    configure_vllm_logging() must add a filter to vllm.v1.engine.async_llm
    that suppresses DEBUG-level check_health messages.
    """
    configure_vllm_logging(logging.INFO)

    async_llm_logger = logging.getLogger("vllm.v1.engine.async_llm")
    assert len(async_llm_logger.filters) == 1

    # Simulate a check_health DEBUG record — should be filtered out
    record = logging.LogRecord(
        name="vllm.v1.engine.async_llm",
        level=logging.DEBUG,
        pathname="",
        lineno=0,
        msg="Called check_health.",
        args=(),
        exc_info=None,
    )
    record.funcName = "check_health"
    health_filter = async_llm_logger.filters[0]
    assert isinstance(health_filter, logging.Filter)
    assert not health_filter.filter(record)

    # Same message at WARNING level — should pass through
    record.levelno = logging.WARNING
    assert health_filter.filter(record)


def test_health_check_filter_not_duplicated():
    """
    Calling configure_vllm_logging() multiple times must not accumulate
    duplicate health-check filters.
    """
    configure_vllm_logging(logging.INFO)
    configure_vllm_logging(logging.INFO)
    configure_vllm_logging(logging.INFO)

    async_llm_logger = logging.getLogger("vllm.v1.engine.async_llm")
    assert len(async_llm_logger.filters) == 1
