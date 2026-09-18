# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stability-tier inference for captured surface symbols.

Extractors capture structure, not policy. This module is the one place that
turns code-grounded signals into a stability tier, so the ledger and
``validate`` can treat removing a non-``stable`` symbol as informational rather
than a broken promise.

Signal families, weakest tier wins (a symbol is only as stable as its
least-stable signal):

- **Declared-public gate (Python/Rust)** -- the public-API surfaces require a
  *positive* signal to count as ``stable``. The extractor records
  ``metadata['declared_public']``: ``True`` for a symbol in a module's
  ``__all__`` or a curated ``.pyi`` stub (Rust: re-exported at the crate root),
  ``False`` for a name that is merely reachable (non-underscore, no ``__all__``).
  A ``False`` flag floors the tier at ``experimental`` -- so an internal-utils
  refactor is informational churn, not a wave of broken promises. An *absent*
  flag is treated as declared, which keeps every other surface (env, crd, helm,
  http, metric, config -- public by construction) and any not-yet-migrated
  extractor at the historical ``stable`` default.
- **CRD served version** -- Kubernetes version names encode maturity directly:
  ``v1alpha1`` -> experimental, ``v1beta2`` -> preview, ``v1`` -> stable. Read
  from ``metadata['version']`` so it works for both the version-level symbol and
  every field under it.
- **Name/path keywords** -- ``experimental`` anywhere in the id is experimental;
  ``alpha`` is experimental; ``preview`` / ``beta`` is preview. Matched on
  word-ish boundaries so ``betamax`` or ``alphabet`` never trip it.

:func:`normalize_stability` is the seam ``build_snapshot`` calls: it only ever
*downgrades* (an extractor that already set a non-``stable`` tier is authoritative
and kept), so inference can never silently promise more stability than an
extractor asserted.
"""

from __future__ import annotations

import re

from api_surface.models import STABILITY_TIERS, SurfaceSymbol

# Tier ordering, most stable first. A lower rank is *less* stable; inference
# keeps whichever signal yields the least-stable (highest-rank) tier.
_TIER_RANK = {"stable": 0, "preview": 1, "experimental": 2}

# CRD version suffix -> tier. ``v1alpha1`` / ``v2beta3`` style names.
_CRD_VERSION_RE = re.compile(r"v\d+(?P<phase>alpha|beta)\d*$", re.IGNORECASE)

# Keyword -> tier, matched on non-alphanumeric boundaries inside the id.
_KEYWORD_TIERS = (
    (
        re.compile(r"(?<![a-z0-9])experimental(?![a-z0-9])", re.IGNORECASE),
        "experimental",
    ),
    (re.compile(r"(?<![a-z0-9])alpha(?![a-z0-9])", re.IGNORECASE), "experimental"),
    (re.compile(r"(?<![a-z0-9])preview(?![a-z0-9])", re.IGNORECASE), "preview"),
    (re.compile(r"(?<![a-z0-9])beta(?![a-z0-9])", re.IGNORECASE), "preview"),
)


def _least_stable(a: str, b: str) -> str:
    """Return the less-stable of two tiers (higher rank wins)."""
    return a if _TIER_RANK.get(a, 0) >= _TIER_RANK.get(b, 0) else b


def _crd_version_tier(version: str) -> str:
    """Map a CRD version name to a stability tier (``stable`` when mature)."""
    match = _CRD_VERSION_RE.match(version.strip())
    if not match:
        return "stable"
    return "experimental" if match.group("phase").lower() == "alpha" else "preview"


# Surfaces that require a positive ``declared_public`` signal to be ``stable``.
# Every other surface (env, crd, helm, http, metric, config) is public by
# construction -- its symbols *are* the user contract -- so it keeps the
# historical ``stable`` default and an absent flag never demotes it.
_DECLARED_PUBLIC_SURFACES = frozenset({"python", "rust"})


def infer_stability(symbol: SurfaceSymbol) -> str:
    """Infer a stability tier for ``symbol`` from its signals.

    Returns the least-stable tier implied by any signal, or ``stable`` when no
    signal fires. Pure and side-effect free; callers decide whether to apply it.
    """
    tier = "stable"

    # Declared-public gate: a python/rust symbol the extractor flagged as not
    # declared public (reachable but absent from __all__ / not re-exported) is
    # floored at experimental. Absent flag => treated as declared (stable base).
    if (
        symbol.surface in _DECLARED_PUBLIC_SURFACES
        and symbol.metadata.get("declared_public") is False
    ):
        tier = _least_stable(tier, "experimental")

    if symbol.surface == "crd":
        version = str(symbol.metadata.get("version", ""))
        if version:
            tier = _least_stable(tier, _crd_version_tier(version))

    for pattern, keyword_tier in _KEYWORD_TIERS:
        if pattern.search(symbol.id):
            tier = _least_stable(tier, keyword_tier)

    return tier


def normalize_stability(symbol: SurfaceSymbol) -> None:
    """Downgrade ``symbol.stability`` in place from inferred signals.

    Only ever lowers stability: a symbol an extractor already marked
    ``preview`` / ``experimental`` is left alone (the extractor is
    authoritative), and an unknown stored tier is treated as ``stable`` for the
    comparison so inference can still demote it.
    """
    current = symbol.stability if symbol.stability in STABILITY_TIERS else "stable"
    symbol.stability = _least_stable(current, infer_stability(symbol))
