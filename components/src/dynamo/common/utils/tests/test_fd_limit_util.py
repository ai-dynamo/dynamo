# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the default (no target) mode of `raise_fd_limit`, used by the
standalone router and the mocker. Targeted mode is covered through the
frontend wrapper in `dynamo/frontend/tests/test_fd_limit.py`."""

import pytest

from dynamo.common.utils.fd_limit import raise_fd_limit

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]

resource = pytest.importorskip("resource")


def _record_setrlimit(monkeypatch) -> dict:
    captured: dict = {}
    monkeypatch.setattr(
        resource, "setrlimit", lambda _res, limits: captured.update(limits=limits)
    )
    return captured


def test_default_raises_soft_to_hard(monkeypatch):
    monkeypatch.setattr(resource, "getrlimit", lambda _res: (1024, 1_048_576))
    captured = _record_setrlimit(monkeypatch)
    raise_fd_limit()
    assert captured["limits"] == (1_048_576, 1_048_576)


@pytest.mark.parametrize(
    "limits",
    [
        (1_048_576, 1_048_576),  # already at the hard limit
        (1024, resource.RLIM_INFINITY),  # unbounded hard limit: no finite target
    ],
)
def test_default_leaves_limit_unchanged(monkeypatch, limits):
    monkeypatch.setattr(resource, "getrlimit", lambda _res: limits)
    captured = _record_setrlimit(monkeypatch)
    raise_fd_limit()
    assert "limits" not in captured
