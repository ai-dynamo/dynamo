# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``SpiffeJwtAuth`` — validates SPIFFE JWT-SVIDs against a SPIRE JWKS
(DEP-XXXX PR 3.5 follow-up).

Flow per ``validate(token)``:

1. Verify the JWT signature using the JWKS fetched at init (the library
   picks the key whose ``kid`` matches the JWT header).
2. Verify the ``aud`` claim matches ``self._audience``.
3. Extract the ``sub`` claim, verify it is a SPIFFE ID whose trust
   domain matches ``self._trust_domain``, then verify the full SPIFFE ID
   is in the ``trusted_spiffe_ids`` allow-list.

Key design choices:

- **Signature + audience are verified by PyJWT** rather than by hand.
  ``audience=self._audience`` is passed to ``jwt.decode`` so a missing
  or wrong ``aud`` claim raises before we ever look at ``sub``.
- **JWKS fetched once at ``__init__``**, using ``PyJWKClient``; SPIRE
  rotation windows (default 24h) are long enough that startup-time
  fetch is fine for v1. A hot-reload variant is a follow-up if/when
  operators report "plugin register fails right after a SPIRE rotation"
  incidents.
- **Algorithms** default to ``["RS256", "ES256"]`` — the two SVID
  signing algorithms SPIRE actually emits. Callers can override for
  test envs or if SPIRE is configured differently.
- **Trust-domain check** is explicit (not left to the allow-list),
  because SPIFFE IDs under other trust domains will usually look like
  other deployments' IDs — rejecting them with a clear "wrong trust
  domain" error is easier to diagnose than "not in allow-list".
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

from dynamo.planner.plugins.registry.auth.base import (
    AuthIdentity,
    AuthValidator,
)
from dynamo.planner.plugins.registry.errors import AuthError

log = logging.getLogger(__name__)

_SPIFFE_SCHEME = "spiffe://"
_DEFAULT_ALGORITHMS = ("RS256", "ES256")


class SpiffeJwtAuth(AuthValidator):
    """Validate SPIFFE JWT-SVIDs issued by SPIRE.

    Args:
        jwks_endpoint: URL of the SPIRE JWKS bundle (e.g.
            ``https://spire-server.<ns>.svc/keys``). Fetched once at
            init; see module docstring for rationale.
        audience: expected JWT ``aud`` claim value.
        trust_domain: SPIFFE trust domain (e.g. ``example.com``); the
            ``sub`` claim must be ``spiffe://<trust_domain>/...``.
        trusted_spiffe_ids: iterable of full SPIFFE IDs (e.g.
            ``spiffe://example.com/planner-plugin``). Empty allow-list
            rejects everything.
        algorithms: override the JWT algorithms accepted (default:
            ``RS256`` and ``ES256``, matching SPIRE).
        jwk_client: inject a pre-built ``PyJWKClient`` to skip the
            network fetch at init (used by tests).
    """

    def __init__(
        self,
        jwks_endpoint: str,
        audience: str,
        trust_domain: str,
        trusted_spiffe_ids: Iterable[str],
        *,
        algorithms: Optional[Iterable[str]] = None,
        jwk_client: Optional[object] = None,
    ) -> None:
        if not audience:
            raise ValueError("SpiffeJwtAuth: audience must be non-empty")
        if not trust_domain:
            raise ValueError("SpiffeJwtAuth: trust_domain must be non-empty")
        self._endpoint = jwks_endpoint
        self._audience = audience
        self._trust_domain = trust_domain
        self._trusted: frozenset[str] = frozenset(trusted_spiffe_ids)
        self._algorithms = list(algorithms) if algorithms else list(_DEFAULT_ALGORITHMS)
        if not self._trusted:
            log.warning(
                "SpiffeJwtAuth: trusted_spiffe_ids is empty; every valid "
                "JWT will still be rejected by the allow-list (fail-closed)."
            )
        # Defer import so the dependency is only required when the
        # validator is actually constructed (build_auth_validator path).
        if jwk_client is not None:
            self._jwk_client = jwk_client
        else:
            from jwt import PyJWKClient

            self._jwk_client = PyJWKClient(jwks_endpoint)

    async def validate(self, token: str) -> AuthIdentity:
        if not token:
            raise AuthError("spiffe_jwt: empty token")

        import jwt as pyjwt

        try:
            signing_key = self._jwk_client.get_signing_key_from_jwt(token)
            key = getattr(signing_key, "key", signing_key)
            claims = pyjwt.decode(
                token,
                key=key,
                algorithms=self._algorithms,
                audience=self._audience,
            )
        except pyjwt.InvalidTokenError as exc:
            # Includes signature / audience / expiry / malformed JWT.
            raise AuthError(f"spiffe_jwt: JWT rejected ({exc})") from exc
        except Exception as exc:
            # JWKS fetch transient errors or unexpected key-resolution
            # failures. Don't leak the underlying URL / HTTP body.
            log.info("spiffe_jwt: key resolution failed: %s", exc)
            raise AuthError("spiffe_jwt: key resolution failed") from exc

        sub = claims.get("sub")
        if not isinstance(sub, str) or not sub.startswith(_SPIFFE_SCHEME):
            raise AuthError(
                f"spiffe_jwt: sub claim {sub!r} is not a SPIFFE ID"
            )
        after_scheme = sub[len(_SPIFFE_SCHEME):]
        td, sep, _ = after_scheme.partition("/")
        if not sep or td != self._trust_domain:
            raise AuthError(
                f"spiffe_jwt: trust domain {td!r} does not match "
                f"configured {self._trust_domain!r}"
            )
        if sub not in self._trusted:
            raise AuthError(
                f"spiffe_jwt: SPIFFE ID {sub!r} not in allow-list"
            )
        return AuthIdentity(
            source="spiffe_jwt",
            subject=sub,
            metadata={"audience": self._audience},
        )


__all__ = ["SpiffeJwtAuth"]
