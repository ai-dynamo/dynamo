# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from tests.utils import decision_api as client

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


def test_systemone_client_retries_only_bounded_capacity_errors(monkeypatch):
    sleeps = []
    monkeypatch.setattr(client.time, "sleep", sleeps.append)
    responses = iter(
        SimpleNamespace(status_code=status, headers={"retry-after": "1"}, text="")
        for status in [429, 429, 200]
    )
    monkeypatch.setattr(
        client.requests, "post", lambda *args, **kwargs: next(responses)
    )
    assert client.post_with_capacity_retry("unused", {}).status_code == 200
    assert sleeps == [1, 1]
    overloaded = SimpleNamespace(status_code=429, headers={"retry-after": "1"}, text="")
    responses = iter([overloaded, overloaded, overloaded])
    monkeypatch.setattr(
        client.requests, "post", lambda *args, **kwargs: next(responses)
    )
    assert client.post_with_capacity_retry("unused", {}).status_code == 429
    assert sleeps == [1, 1, 1, 1]
    unavailable = SimpleNamespace(
        status_code=429, headers={}, text="missing retry contract"
    )
    monkeypatch.setattr(client.requests, "post", lambda *args, **kwargs: unavailable)
    with pytest.raises(AssertionError, match="missing retry contract"):
        client.post_with_capacity_retry("unused", {})
    unavailable = SimpleNamespace(
        status_code=503, headers={}, text="worker unavailable"
    )
    monkeypatch.setattr(client.requests, "post", lambda *args, **kwargs: unavailable)
    assert client.post_with_capacity_retry("unused", {}) is unavailable
