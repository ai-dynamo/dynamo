# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for K8sSATokenAuth.

Tests use a fake ``kube_client`` that returns canned TokenReview
responses (dict shape). ``K8sSATokenAuth._get`` handles both real
kubernetes-client models and plain dicts, so these fixtures exercise
the full validation logic without needing the ``kubernetes`` library
or any cluster access.
"""

from __future__ import annotations

import pytest

from dynamo.planner.plugins.registry.auth import K8sSATokenAuth
from dynamo.planner.plugins.registry.errors import AuthError

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


class _FakeKubeClient:
    """Minimal stand-in for ``kubernetes.client.AuthenticationV1Api``.

    ``response`` is the dict returned from ``create_token_review``;
    ``raise_exc`` (if set) is raised instead — simulating a transient
    API server error.
    """

    def __init__(self, response=None, raise_exc=None):
        self.response = response
        self.raise_exc = raise_exc
        self.last_body = None

    def create_token_review(self, body):
        self.last_body = body
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.response


def _ok_review(username, audiences, authenticated=True):
    return {
        "status": {
            "authenticated": authenticated,
            "audiences": audiences,
            "user": {"username": username},
        }
    }


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_accepts_token_when_sa_in_allowlist_and_audience_echoed():
    client = _FakeKubeClient(
        response=_ok_review(
            username="system:serviceaccount:dyno:planner-plugin",
            audiences=["dynamo-planner"],
        )
    )
    auth = K8sSATokenAuth(
        kube_client=client,
        audience="dynamo-planner",
        trusted_service_accounts=["dyno/planner-plugin", "other/ignored"],
    )
    identity = await auth.validate("fake-jwt-token")
    assert identity.source == "k8s_sa"
    assert identity.subject == "dyno/planner-plugin"
    assert identity.metadata["audience"] == "dynamo-planner"


# ---------------------------------------------------------------------------
# Rejections
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rejects_empty_token_without_api_call():
    client = _FakeKubeClient(response=_ok_review("system:serviceaccount:a/b", ["x"]))
    auth = K8sSATokenAuth(
        kube_client=client,
        audience="dynamo-planner",
        trusted_service_accounts=["a/b"],
    )
    with pytest.raises(AuthError, match="empty token"):
        await auth.validate("")
    assert client.last_body is None


@pytest.mark.asyncio
async def test_rejects_when_api_says_not_authenticated():
    client = _FakeKubeClient(
        response={"status": {"authenticated": False, "error": "bad signature"}}
    )
    auth = K8sSATokenAuth(
        kube_client=client,
        audience="dynamo-planner",
        trusted_service_accounts=["dyno/planner-plugin"],
    )
    with pytest.raises(AuthError, match="TokenReview rejected"):
        await auth.validate("junk")


@pytest.mark.asyncio
async def test_rejects_when_audience_not_echoed():
    # Token was issued for a different service; API server authenticated
    # it but didn't echo our audience. Reject defensively.
    client = _FakeKubeClient(
        response=_ok_review(
            username="system:serviceaccount:dyno:planner-plugin",
            audiences=["some-other-service"],
        )
    )
    auth = K8sSATokenAuth(
        kube_client=client,
        audience="dynamo-planner",
        trusted_service_accounts=["dyno/planner-plugin"],
    )
    with pytest.raises(AuthError, match="missing expected audience"):
        await auth.validate("cross-audience-token")


@pytest.mark.asyncio
async def test_rejects_when_sa_not_in_allowlist():
    client = _FakeKubeClient(
        response=_ok_review(
            username="system:serviceaccount:dyno:intruder",
            audiences=["dynamo-planner"],
        )
    )
    auth = K8sSATokenAuth(
        kube_client=client,
        audience="dynamo-planner",
        trusted_service_accounts=["dyno/planner-plugin"],
    )
    with pytest.raises(AuthError, match="not in allow-list"):
        await auth.validate("authenticated-but-unlisted")


@pytest.mark.asyncio
async def test_rejects_non_serviceaccount_user():
    # A real human user authenticated via OIDC — API server authenticates
    # the token but `username` won't have the SA prefix.
    client = _FakeKubeClient(
        response=_ok_review(
            username="alice@example.com",
            audiences=["dynamo-planner"],
        )
    )
    auth = K8sSATokenAuth(
        kube_client=client,
        audience="dynamo-planner",
        trusted_service_accounts=["dyno/planner-plugin"],
    )
    with pytest.raises(AuthError, match="expected system:serviceaccount"):
        await auth.validate("human-token")


@pytest.mark.asyncio
async def test_rejects_malformed_serviceaccount_username():
    client = _FakeKubeClient(
        response=_ok_review(
            username="system:serviceaccount:no-sa-name",
            audiences=["dynamo-planner"],
        )
    )
    auth = K8sSATokenAuth(
        kube_client=client,
        audience="dynamo-planner",
        trusted_service_accounts=["dyno/planner-plugin"],
    )
    with pytest.raises(AuthError, match="malformed SA username"):
        await auth.validate("malformed-token")


@pytest.mark.asyncio
async def test_rejects_when_api_call_raises():
    class _Boom(Exception):
        pass

    client = _FakeKubeClient(raise_exc=_Boom("api gateway 503"))
    auth = K8sSATokenAuth(
        kube_client=client,
        audience="dynamo-planner",
        trusted_service_accounts=["dyno/planner-plugin"],
    )
    with pytest.raises(AuthError, match="TokenReview API call failed"):
        await auth.validate("any-token")


# ---------------------------------------------------------------------------
# Config-time validation
# ---------------------------------------------------------------------------


def test_empty_audience_rejected_at_construction():
    with pytest.raises(ValueError, match="audience must be non-empty"):
        K8sSATokenAuth(
            kube_client=_FakeKubeClient(),
            audience="",
            trusted_service_accounts=["a/b"],
        )


def test_empty_allowlist_warns_but_constructs(caplog):
    with caplog.at_level("WARNING"):
        auth = K8sSATokenAuth(
            kube_client=_FakeKubeClient(),
            audience="dynamo-planner",
            trusted_service_accounts=[],
        )
    assert auth is not None
    assert any("empty" in rec.getMessage().lower() for rec in caplog.records)


# ---------------------------------------------------------------------------
# Body construction — exercises the _call_token_review path to make sure
# we actually request the expected audience.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_request_body_includes_configured_audience():
    client = _FakeKubeClient(
        response=_ok_review(
            username="system:serviceaccount:dyno:planner-plugin",
            audiences=["dynamo-planner"],
        )
    )
    auth = K8sSATokenAuth(
        kube_client=client,
        audience="dynamo-planner",
        trusted_service_accounts=["dyno/planner-plugin"],
    )
    await auth.validate("fake-jwt-token")
    # last_body is a V1TokenReview; exercise attribute access without
    # hard-coding the kubernetes-client class name.
    body = client.last_body
    assert body is not None
    spec = getattr(body, "spec", None)
    assert spec is not None
    assert spec.token == "fake-jwt-token"
    assert spec.audiences == ["dynamo-planner"]
