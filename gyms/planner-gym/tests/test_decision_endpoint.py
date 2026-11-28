# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest
from autoscaling_arena.decision_endpoint import decision_base_url, decision_http_client

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


@pytest.mark.parametrize(
    "changes",
    [
        {"base_url": None},
        {"base_url": "https://api.typesafe.ai"},
        {"base_url": "http://name:secret@localhost:8123"},
        {"base_url": "http://localhost:8123/v1"},
        {"base_url": "http://localhost:8123?token=secret"},
        {"base_url": "file:///tmp/model"},
        {"base_url": "http://localhost:bad"},
        {"base_url": "http://localhost:0"},
        {"model": "jev-1.13.0"},
        {"provider": "unknown"},
        {"api_key_env": "TYPESAFE_API_KEY"},
        {"api_key_env": "secret-value"},
        # A bearer token must not travel in clear text beyond loopback.
        {"base_url": "http://decision.internal:8123", "api_key_env": "MODEL_TOKEN"},
        {"base_url": "http://10.0.0.7:8123", "api_key_env": "MODEL_TOKEN"},
    ],
)
def test_self_hosted_requires_explicit_independent_destination(changes):
    config = dict(
        provider="self_hosted", base_url="http://localhost:8123", model="open-model"
    )
    config.update(changes)
    with pytest.raises(ValueError):
        decision_base_url(**config)


@pytest.mark.parametrize(
    ("base_url", "api_key_env"),
    [
        ("https://decision.internal:8123", "MODEL_TOKEN"),
        ("http://localhost:8123", "MODEL_TOKEN"),
        ("http://127.0.0.1:8123", "MODEL_TOKEN"),
        ("http://[::1]:8123", "MODEL_TOKEN"),
        # No credential is sent, so a private plain-HTTP endpoint stays usable.
        ("http://decision.internal:8123", None),
    ],
)
def test_token_is_sent_only_over_https_or_loopback(base_url, api_key_env):
    origin = decision_base_url(
        provider="self_hosted",
        base_url=base_url,
        model="open-model",
        api_key_env=api_key_env,
    )
    assert origin == base_url


def test_hosted_credentials_cannot_be_redirected():
    with pytest.raises(ValueError, match="fixed endpoint"):
        decision_base_url(base_url="http://localhost:8123")


@pytest.mark.parametrize("use_key", [False, True])
def test_self_hosted_request_never_carries_typesafe_key(monkeypatch, use_key):
    # The HTTP client ships with the optional `jev` extra.
    httpx = pytest.importorskip("httpx")
    monkeypatch.setenv("TYPESAFE_API_KEY", "paid-key-must-not-be-forwarded")
    monkeypatch.setenv("DECISION_MODEL_TOKEN", "local-token")
    actual_client = httpx.AsyncClient
    seen = []

    def handle(request):
        seen.append(request)
        return httpx.Response(200, json={"model": "open-model"})

    def client_with_transport(**kwargs):
        assert kwargs["follow_redirects"] is False
        assert kwargs["trust_env"] is False
        return actual_client(transport=httpx.MockTransport(handle), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client_with_transport)
    client = decision_http_client(
        provider="self_hosted",
        base_url="http://localhost:8123/",
        model="open-model",
        timeout_s=5,
        api_key_env="DECISION_MODEL_TOKEN" if use_key else None,
    )

    async def run():
        async with client:
            await client.post("/v1/systemone", json={"model": "open-model"})

    asyncio.run(run())
    assert str(seen[0].url) == "http://localhost:8123/v1/systemone"
    assert seen[0].headers.get("Authorization") == (
        "Bearer local-token" if use_key else None
    )


def test_explicit_auth_does_not_silently_fall_back(monkeypatch):
    monkeypatch.delenv("DECISION_MODEL_TOKEN", raising=False)
    with pytest.raises(ValueError, match="requires DECISION_MODEL_TOKEN"):
        decision_http_client(
            provider="self_hosted",
            base_url="http://localhost:8123",
            model="open-model",
            timeout_s=5,
            api_key_env="DECISION_MODEL_TOKEN",
        )
