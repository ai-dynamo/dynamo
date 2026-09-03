# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Platform-safety tests for the frontend fd-limit hardening (`_raise_fd_limit`).

`resource` is a Unix-only stdlib module, so the hardening must degrade to a
no-op on platforms without it (e.g. Windows) and must never break startup.
"""

import builtins

import pytest

from dynamo.frontend.main import FRONTEND_FD_LIMIT_TARGET, _raise_fd_limit

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def test_noop_when_resource_missing(monkeypatch):
    """Simulate Windows: `import resource` fails -> clean no-op, not ImportError."""
    real_import = builtins.__import__

    def _no_resource(name, *args, **kwargs):
        if name == "resource":
            raise ImportError("simulated Windows: no `resource` module")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_resource)
    _raise_fd_limit()  # must return cleanly


def test_swallows_setrlimit_failure(monkeypatch):
    """A denied setrlimit (restricted environment) is swallowed best-effort."""
    resource = pytest.importorskip("resource")
    monkeypatch.setattr(resource, "getrlimit", lambda _res: (64, 1_000_000))

    def _deny(*args, **kwargs):
        raise ValueError("simulated: not permitted to raise the limit")

    monkeypatch.setattr(resource, "setrlimit", _deny)
    _raise_fd_limit()  # must return cleanly despite setrlimit raising


def test_raises_limit_on_unix(monkeypatch):
    """On Unix, the soft limit is lifted toward the target (bounded by hard)."""
    resource = pytest.importorskip("resource")
    captured = {}
    monkeypatch.setattr(resource, "getrlimit", lambda _res: (64, 1_000_000))
    monkeypatch.setattr(
        resource, "setrlimit", lambda _res, limits: captured.update(limits=limits)
    )
    _raise_fd_limit(target=4096)
    assert captured["limits"] == (4096, 1_000_000)


def test_noop_when_already_sufficient(monkeypatch):
    """No setrlimit call when the soft limit already meets the target."""
    resource = pytest.importorskip("resource")
    monkeypatch.setattr(
        resource, "getrlimit", lambda _res: (FRONTEND_FD_LIMIT_TARGET, 1_000_000)
    )

    def _fail(*args, **kwargs):
        raise AssertionError("setrlimit must not be called when already sufficient")

    monkeypatch.setattr(resource, "setrlimit", _fail)
    _raise_fd_limit()  # soft already == target -> no setrlimit
