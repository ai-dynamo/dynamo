# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``K8sSATokenAuth`` — validates Kubernetes ServiceAccount projected tokens
via the cluster's ``TokenReview`` API.

Flow per ``validate(token)``:

1. POST a ``TokenReview`` with ``spec.token=token`` and
   ``spec.audiences=[self._audience]``. The API server validates the
   signature against the cluster's signing keys and rejects tokens
   whose audience list doesn't include ours — so we don't need a JWKS
   fetch or claim-by-claim parser on our side.
2. If ``status.authenticated`` is false, reject.
3. Extract the SA from ``status.user.username`` (format
   ``system:serviceaccount:<ns>:<sa>``). Normalise to ``<ns>/<sa>``.
4. Reject if the normalised SA is not in the ``trusted_service_accounts``
   allow-list.

Design notes:

- The ``kubernetes`` client is sync; we wrap each ``TokenReview`` call in
  ``asyncio.to_thread`` so the orchestrator event loop is never blocked.
  The async ``kubernetes_asyncio`` variant would let us skip the thread
  hop but requires a full async client context per call, which is more
  machinery than this low-QPS path warrants.
- The ``kube_client`` is injected (not constructed inside ``__init__``)
  so tests can substitute a fake without ``kubeconfig`` / RBAC / cluster
  access, and so callers can reuse a long-lived in-cluster config.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Iterable

from dynamo.planner.plugins.registry.auth.base import (
    AuthIdentity,
    AuthValidator,
)
from dynamo.planner.plugins.registry.errors import AuthError

log = logging.getLogger(__name__)

_SA_USERNAME_PREFIX = "system:serviceaccount:"


class K8sSATokenAuth(AuthValidator):
    """Validate bearer tokens via Kubernetes ``TokenReview``.

    Args:
        kube_client: a ``kubernetes.client.AuthenticationV1Api`` instance
            (or any object exposing ``create_token_review(body)`` with the
            same response shape). Must be safe to call from a worker
            thread.
        audience: the audience string the Planner expects in projected
            SA tokens. The API server rejects tokens whose audience list
            doesn't include this value.
        trusted_service_accounts: iterable of ``namespace/serviceaccount``
            strings. An empty allow-list rejects every authenticated
            caller (fail-closed).
    """

    def __init__(
        self,
        kube_client: Any,
        audience: str,
        trusted_service_accounts: Iterable[str],
    ) -> None:
        if not audience:
            raise ValueError("K8sSATokenAuth: audience must be non-empty")
        self._client = kube_client
        self._audience = audience
        self._trusted: frozenset[str] = frozenset(trusted_service_accounts)
        if not self._trusted:
            log.warning(
                "K8sSATokenAuth: trusted_service_accounts is empty; every "
                "authenticated caller will be rejected (fail-closed)."
            )

    async def validate(self, token: str) -> AuthIdentity:
        if not token:
            raise AuthError("k8s_sa: empty token")

        review = await asyncio.to_thread(self._call_token_review, token)
        status = _get(review, "status")
        if status is None or not _get(status, "authenticated"):
            reason = _get(status, "error") if status is not None else None
            raise AuthError(f"k8s_sa: TokenReview rejected ({reason or 'not authenticated'})")

        # Defence in depth: TokenReview returns the subset of our
        # requested audiences that actually appeared in the token. If
        # ours isn't echoed back, the token was issued for a different
        # service — refuse it even though the API server authenticated.
        echoed_audiences = _get(status, "audiences") or []
        if self._audience not in echoed_audiences:
            raise AuthError(
                f"k8s_sa: token missing expected audience {self._audience!r}"
            )

        user = _get(status, "user")
        username = _get(user, "username") if user is not None else None
        if not username or not username.startswith(_SA_USERNAME_PREFIX):
            raise AuthError(
                f"k8s_sa: unexpected user.username {username!r} "
                "(expected system:serviceaccount:<ns>:<sa>)"
            )
        sa_path = username[len(_SA_USERNAME_PREFIX):]  # "<ns>:<sa>"
        ns, sep, sa_name = sa_path.partition(":")
        if not sep or not ns or not sa_name:
            raise AuthError(f"k8s_sa: malformed SA username {username!r}")
        sa_key = f"{ns}/{sa_name}"
        if sa_key not in self._trusted:
            raise AuthError(
                f"k8s_sa: service account {sa_key!r} not in allow-list"
            )
        return AuthIdentity(
            source="k8s_sa",
            subject=sa_key,
            metadata={"audience": self._audience},
        )

    def _call_token_review(self, token: str) -> Any:
        """Invoke the sync ``TokenReview`` API. Split out so tests can
        inject a fake and so the call site is a single ``to_thread`` hop.
        """
        from kubernetes.client import V1TokenReview, V1TokenReviewSpec

        body = V1TokenReview(
            spec=V1TokenReviewSpec(token=token, audiences=[self._audience])
        )
        try:
            return self._client.create_token_review(body=body)
        except Exception as exc:
            # Don't leak API error bodies (may include request ID or
            # policy fragments) — just log and surface a generic reject.
            log.info("k8s_sa: TokenReview API call failed: %s", exc)
            raise AuthError("k8s_sa: TokenReview API call failed") from exc


def _get(obj: Any, attr: str) -> Any:
    """Uniformly read from either a kubernetes-client model (attributes)
    or a plain ``dict`` (keys). Lets tests pass ``dict`` fixtures without
    mocking the entire ``kubernetes.client`` model hierarchy."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(attr)
    return getattr(obj, attr, None)


__all__ = ["K8sSATokenAuth"]
