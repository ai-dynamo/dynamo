# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dynamo.common.utils.env."""

import logging

import pytest

import dynamo.common.utils.env as env_module
from dynamo.common.utils.env import env_bool, fpm_trace_enabled

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


class TestEnvBool:
    def test_unset_returns_default(self, monkeypatch):
        monkeypatch.delenv("FOO", raising=False)
        assert env_bool("FOO") is False
        assert env_bool("FOO", default=True) is True

    def test_empty_returns_default(self, monkeypatch):
        monkeypatch.setenv("FOO", "")
        assert env_bool("FOO") is False
        assert env_bool("FOO", default=True) is True

    @pytest.mark.parametrize("value", ["true", "True", "TRUE", "1", "yes", "YES"])
    def test_truthy_values(self, monkeypatch, value):
        monkeypatch.setenv("FOO", value)
        assert env_bool("FOO") is True

    @pytest.mark.parametrize("value", ["false", "0", "no", "on", "off", "anything"])
    def test_falsy_values(self, monkeypatch, value):
        monkeypatch.setenv("FOO", value)
        assert env_bool("FOO") is False


class TestFpmTraceEnabled:
    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "on", "ON", "yes", "YES"])
    def test_truthy_values(self, monkeypatch, value):
        monkeypatch.setenv("DYN_FPM_TRACE", value)
        assert fpm_trace_enabled() is True

    @pytest.mark.parametrize("value", ["0", "false", "FALSE", "off", "OFF", "no", "NO"])
    def test_falsy_values(self, monkeypatch, value):
        monkeypatch.setenv("DYN_FPM_TRACE", value)
        assert fpm_trace_enabled() is False

    def test_invalid_value_warns_once_and_disables(self, monkeypatch, caplog):
        monkeypatch.setattr(env_module, "_fpm_trace_invalid_warning_emitted", False)
        monkeypatch.setenv("DYN_FPM_TRACE", "sometimes")

        with caplog.at_level(logging.WARNING, logger=env_module.__name__):
            assert fpm_trace_enabled() is False
            assert fpm_trace_enabled() is False

        assert caplog.text.count("Invalid DYN_FPM_TRACE value") == 1
        assert "FPM tracing is disabled for this worker" in caplog.text
