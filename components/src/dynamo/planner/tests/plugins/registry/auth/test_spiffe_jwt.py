# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SpiffeJwtAuth (DEP-XXXX PR 3.5 follow-up).

Uses a real ``cryptography`` RSA keypair + hand-rolled ``PyJWT`` tokens
instead of a full SPIRE deployment. The JWKS client is substituted via
the ``jwk_client`` kwarg so no HTTP fetch happens.
"""

from __future__ import annotations

import time

import pytest

jwt = pytest.importorskip("jwt")
_crypto = pytest.importorskip("cryptography.hazmat.primitives.asymmetric.rsa")

from dynamo.planner.plugins.registry.auth import SpiffeJwtAuth
from dynamo.planner.plugins.registry.errors import AuthError

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


# ---------------------------------------------------------------------------
# Test fixtures: RSA keypair + fake JWK client
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rsa_keys():
    """Generate a throwaway RSA keypair for signing test JWTs."""
    from cryptography.hazmat.primitives.asymmetric import rsa

    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public = private.public_key()
    return private, public


class _FakeJwkClient:
    """Stand-in for ``jwt.PyJWKClient`` — returns the same public key
    for every ``get_signing_key_from_jwt`` call, matching the one-kid
    SPIRE configuration used in these tests."""

    def __init__(self, public_key):
        self._key = public_key

    def get_signing_key_from_jwt(self, token):
        class _SigningKey:
            key = self._key

        return _SigningKey()


class _AlwaysRaisingJwkClient:
    """Simulates JWKS fetch / key resolution failure."""

    def get_signing_key_from_jwt(self, token):
        raise RuntimeError("jwks endpoint 503")


def _sign(private_key, claims, *, algorithm="RS256"):
    return jwt.encode(claims, private_key, algorithm=algorithm)


def _default_claims(
    sub="spiffe://example.com/planner-plugin",
    aud="dynamo-planner",
    exp_offset=300,
):
    return {"sub": sub, "aud": aud, "exp": int(time.time()) + exp_offset}


def _make_auth(public_key, **overrides):
    defaults = dict(
        jwks_endpoint="https://spire.invalid/keys",  # unused; jwk_client wins
        audience="dynamo-planner",
        trust_domain="example.com",
        trusted_spiffe_ids=["spiffe://example.com/planner-plugin"],
        jwk_client=_FakeJwkClient(public_key),
    )
    defaults.update(overrides)
    return SpiffeJwtAuth(**defaults)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_accepts_valid_jwt_with_trusted_spiffe_id(rsa_keys):
    private, public = rsa_keys
    token = _sign(private, _default_claims())

    auth = _make_auth(public)
    identity = await auth.validate(token)

    assert identity.source == "spiffe_jwt"
    assert identity.subject == "spiffe://example.com/planner-plugin"
    assert identity.metadata["audience"] == "dynamo-planner"


# ---------------------------------------------------------------------------
# Rejections — JWT-layer (PyJWT raises InvalidTokenError)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rejects_empty_token_without_key_lookup(rsa_keys):
    _, public = rsa_keys
    auth = _make_auth(public)
    with pytest.raises(AuthError, match="empty token"):
        await auth.validate("")


@pytest.mark.asyncio
async def test_rejects_wrong_audience(rsa_keys):
    private, public = rsa_keys
    token = _sign(private, _default_claims(aud="some-other-service"))

    auth = _make_auth(public)
    with pytest.raises(AuthError, match="JWT rejected"):
        await auth.validate(token)


@pytest.mark.asyncio
async def test_rejects_expired_token(rsa_keys):
    private, public = rsa_keys
    token = _sign(private, _default_claims(exp_offset=-60))

    auth = _make_auth(public)
    with pytest.raises(AuthError, match="JWT rejected"):
        await auth.validate(token)


@pytest.mark.asyncio
async def test_rejects_wrong_signature(rsa_keys):
    from cryptography.hazmat.primitives.asymmetric import rsa

    private, _ = rsa_keys
    # Different public key → signature will not verify.
    wrong_public = rsa.generate_private_key(
        public_exponent=65537, key_size=2048
    ).public_key()

    token = _sign(private, _default_claims())
    auth = _make_auth(wrong_public)
    with pytest.raises(AuthError, match="JWT rejected"):
        await auth.validate(token)


# ---------------------------------------------------------------------------
# Rejections — SPIFFE layer (our own post-JWT checks)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rejects_non_spiffe_sub(rsa_keys):
    private, public = rsa_keys
    token = _sign(private, _default_claims(sub="alice@example.com"))

    auth = _make_auth(public)
    with pytest.raises(AuthError, match="not a SPIFFE ID"):
        await auth.validate(token)


@pytest.mark.asyncio
async def test_rejects_wrong_trust_domain(rsa_keys):
    private, public = rsa_keys
    token = _sign(
        private, _default_claims(sub="spiffe://other-domain.com/planner-plugin")
    )

    auth = _make_auth(public)
    with pytest.raises(AuthError, match="trust domain"):
        await auth.validate(token)


@pytest.mark.asyncio
async def test_rejects_spiffe_id_not_in_allowlist(rsa_keys):
    private, public = rsa_keys
    token = _sign(
        private,
        _default_claims(sub="spiffe://example.com/someone-else"),
    )

    auth = _make_auth(public)
    with pytest.raises(AuthError, match="not in allow-list"):
        await auth.validate(token)


# ---------------------------------------------------------------------------
# Rejections — infra (JWKS fetch / key resolution)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rejects_when_jwks_client_raises():
    auth = SpiffeJwtAuth(
        jwks_endpoint="https://spire.invalid/keys",
        audience="dynamo-planner",
        trust_domain="example.com",
        trusted_spiffe_ids=["spiffe://example.com/planner-plugin"],
        jwk_client=_AlwaysRaisingJwkClient(),
    )
    with pytest.raises(AuthError, match="key resolution failed"):
        await auth.validate("any-token")


# ---------------------------------------------------------------------------
# Config-time validation
# ---------------------------------------------------------------------------


def test_empty_audience_rejected_at_construction():
    with pytest.raises(ValueError, match="audience must be non-empty"):
        SpiffeJwtAuth(
            jwks_endpoint="https://spire.invalid/keys",
            audience="",
            trust_domain="example.com",
            trusted_spiffe_ids=["spiffe://example.com/planner-plugin"],
            jwk_client=_FakeJwkClient(object()),
        )


def test_empty_trust_domain_rejected_at_construction():
    with pytest.raises(ValueError, match="trust_domain must be non-empty"):
        SpiffeJwtAuth(
            jwks_endpoint="https://spire.invalid/keys",
            audience="dynamo-planner",
            trust_domain="",
            trusted_spiffe_ids=["spiffe://example.com/planner-plugin"],
            jwk_client=_FakeJwkClient(object()),
        )


def test_empty_allowlist_warns_but_constructs(caplog, rsa_keys):
    _, public = rsa_keys
    with caplog.at_level("WARNING"):
        auth = SpiffeJwtAuth(
            jwks_endpoint="https://spire.invalid/keys",
            audience="dynamo-planner",
            trust_domain="example.com",
            trusted_spiffe_ids=[],
            jwk_client=_FakeJwkClient(public),
        )
    assert auth is not None
    assert any("empty" in rec.getMessage().lower() for rec in caplog.records)
