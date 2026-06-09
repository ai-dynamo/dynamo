# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dynamo.common.utils.env."""

import pytest

from dynamo.common.utils.env import env_bool

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

    @pytest.mark.parametrize("value", ["false", "0", "no", "anything"])
    def test_falsy_values(self, monkeypatch, value):
        monkeypatch.setenv("FOO", value)
        assert env_bool("FOO") is False
