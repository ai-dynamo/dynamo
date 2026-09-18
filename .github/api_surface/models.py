# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""API surface tracker data models.

Frozen contracts for the code-grounded surface diff engine and the durable
cross-release deprecation ledger. Every extractor, the diff engine, the ledger
merge, the renderers, and the backfill all speak the schemas defined here.

The change vocabulary covers code-grounded additions, removals, incompatible
signature changes, relocations, and native deprecation markers.

Symbol id grammar (stable across releases, one form per surface):

- ``python:<module>.<qualname>[@setter|@deleter]``  e.g. ``python:dynamo.runtime.Config.value@setter``
- ``crd:<Kind>@<version>.<dotpath>``  e.g. ``crd:DynamoGraphDeployment@v1alpha1.spec.services[].replicas``
- ``helm:<chart>:<dotted.key>``  e.g. ``helm:platform:frontend.replicas``
- ``metric:<module>::<CONST>``  e.g. ``metric:frontend_service::REQUESTS_TOTAL``
- ``http:<METHOD> <path>``  e.g. ``http:POST /v1/chat/completions``
- ``env:<VAR>``  e.g. ``env:DYN_LOG``
- ``config:<source>::<struct>.<field>``  e.g. ``config:src/args::RouterConfig.kv_overlap_score_weight``
- ``rust:<crate>::<path>``  e.g. ``rust:dynamo-runtime::component::Endpoint::client``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from api_surface.results import OperationResult

BREAKING_CHANGE_TYPES = {
    "removed_api",
    "deprecated_api",
    "removed_flag",
    "renamed_function",
    "behavioral_change",
    "config_change",
    "env_var_change",
    "dependency_change",
}

# =============================================================================
# Constants
# =============================================================================

# Schema versions for forward-compatible migration of persisted artifacts.
SNAPSHOT_SCHEMA_VERSION = 1
LEDGER_SCHEMA_VERSION = 1

# Tracked surfaces match the policy's public boundary. Rust is included because
# native binding deprecations are marked in Rust even when exposed through the
# Python surface.
SURFACES = {
    "python",
    "crd",
    "helm",
    "metric",
    "http",
    "env",
    "config",
    "rust",
}

# Stability tiers. Removing a non-``stable`` symbol is informational, not a
# broken-promise validation failure.
STABILITY_TIERS = {"stable", "preview", "experimental"}

# Ledger lifecycle states.
LIFECYCLE_STATUSES = {
    "stable",
    "added",
    "deprecated",
    "scheduled_for_removal",
    "removed",
    "relocated",
}

# Review gating: low-confidence sources land ``needs_review`` and are not
# authoritative until ``confirmed``.
REVIEW_STATUSES = {"auto", "needs_review", "confirmed"}

# Where a change/entry came from, in descending authority.
SOURCES = {"diff", "marker", "llm", "label"}

# Change vocabulary: the release-notes breaking-change types plus the
# code-grounded diff events the surface engine emits.
CHANGE_TYPES = BREAKING_CHANGE_TYPES | {
    "added",
    "removed",
    "signature_changed",
    "relocated",
    "served_version_removed",
}


# =============================================================================
# SurfaceSymbol
# =============================================================================


@dataclass
class SurfaceSymbol:
    """A single addressable element of the public API surface.

    Attributes:
        surface: One of ``SURFACES``.
        kind: Surface-specific element kind (e.g. ``function``, ``method``,
            ``class``, ``attribute``, ``field``, ``endpoint``, ``metric``,
            ``value``, ``env_var``, ``crate_item``).
        id: Stable cross-release identifier following the grammar in the
            module docstring.
        signature: Canonicalized signature string (whitespace-normalized,
            defaults standardized) used for change detection.
        component: Project component area derived by the caller when known.
        stability: One of ``STABILITY_TIERS``.
        deprecated: True when a native marker (Rust ``#[deprecated]`` / Python
            ``@deprecated``) is present on the symbol.
        deprecated_note: The marker's note text, when present.
        metadata: Surface-specific extras. CRD symbols carry ``version``,
            ``served``, ``storage``; HTTP symbols carry request/response
            schema fingerprints; metrics carry ``labels``.
    """

    surface: str
    kind: str
    id: str
    signature: str = ""
    component: str = ""
    stability: str = "stable"
    deprecated: bool = False
    deprecated_note: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject malformed symbols before they can weaken policy checks."""
        if self.surface not in SURFACES:
            raise ValueError(f"invalid surface: {self.surface}")
        if self.stability not in STABILITY_TIERS:
            raise ValueError(f"invalid stability tier: {self.stability}")
        if not isinstance(self.deprecated, bool):
            raise ValueError("deprecated must be a boolean")
        if not self.id:
            raise ValueError("symbol id must not be empty")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "surface": self.surface,
            "kind": self.kind,
            "id": self.id,
            "signature": self.signature,
            "component": self.component,
            "stability": self.stability,
            "deprecated": self.deprecated,
            "deprecated_note": self.deprecated_note,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SurfaceSymbol:
        """Deserialize from dictionary."""
        return cls(
            surface=data["surface"],
            kind=data["kind"],
            id=data["id"],
            signature=data.get("signature", ""),
            component=data.get("component", ""),
            stability=data.get("stability", "stable"),
            deprecated=data.get("deprecated", False),
            deprecated_note=data.get("deprecated_note", ""),
            metadata=dict(data.get("metadata", {})),
        )


# =============================================================================
# SurfaceSnapshot
# =============================================================================


@dataclass
class SurfaceSnapshot:
    """The full public surface captured at one release / git ref.

    Attributes:
        release: Release version string (3-dotted, e.g. ``1.2.0``).
        ref: Git ref the snapshot was extracted from (e.g. ``v1.2.0``).
        repo: Source repository identifier.
        generated_at: ISO datetime string of capture time.
        schema_version: ``SNAPSHOT_SCHEMA_VERSION`` at write time.
        coverage: Map of ``surface -> bool``. A surface absent or ``False``
            here was NOT extracted, so the diff engine must never read its
            missing symbols as removals.
        coverage_detail: Optional finer-grained coverage, keyed
            ``<surface>:<unit>`` (e.g. ``rust:dynamo-llm``). Used by the diff
            engine to gate sub-surface units (per-crate rust) so a crate absent
            or renamed at one ref is skipped instead of read as mass removals.
            Distinct from ``coverage`` so the surface-level coverage math is
            never perturbed.
        symbols: The captured symbols. Persisted sorted by ``id``.
    """

    release: str
    ref: str
    repo: str = "ai-dynamo/dynamo"
    generated_at: str = ""
    schema_version: int = SNAPSHOT_SCHEMA_VERSION
    coverage: dict[str, bool] = field(default_factory=dict)
    coverage_detail: dict[str, bool] = field(default_factory=dict)
    symbols: list[SurfaceSymbol] = field(default_factory=list)

    def sorted_symbols(self) -> list[SurfaceSymbol]:
        """Return symbols ordered by ``id`` for deterministic output."""
        return sorted(self.symbols, key=lambda s: s.id)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary with symbols sorted by id."""
        return {
            "schema_version": self.schema_version,
            "release": self.release,
            "ref": self.ref,
            "repo": self.repo,
            "generated_at": self.generated_at,
            "coverage": dict(sorted(self.coverage.items())),
            "coverage_detail": dict(sorted(self.coverage_detail.items())),
            "symbols": [s.to_dict() for s in self.sorted_symbols()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SurfaceSnapshot:
        """Deserialize from dictionary."""
        schema_version = int(data.get("schema_version", SNAPSHOT_SCHEMA_VERSION))
        if schema_version != SNAPSHOT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported snapshot schema {schema_version}; "
                f"expected {SNAPSHOT_SCHEMA_VERSION}"
            )
        return cls(
            release=data["release"],
            ref=data["ref"],
            repo=data.get("repo", "ai-dynamo/dynamo"),
            generated_at=data.get("generated_at", ""),
            schema_version=schema_version,
            coverage=dict(data.get("coverage", {})),
            coverage_detail=dict(data.get("coverage_detail", {})),
            symbols=[SurfaceSymbol.from_dict(s) for s in data.get("symbols", [])],
        )


# =============================================================================
# SurfaceChange
# =============================================================================


@dataclass
class SurfaceChange:
    """One detected change between two snapshots or from a signal source.

    Attributes:
        change_type: One of ``CHANGE_TYPES``.
        id: Symbol id the change applies to.
        surface: One of ``SURFACES``.
        component: Dynamo component area.
        from_signature: Prior signature (empty for additions).
        to_signature: New signature (empty for removals).
        summary: One-sentence human-readable description.
        confidence: ``high`` / ``medium`` / ``low``.
        source: One of ``SOURCES``. Code-grounded diff events are ``diff`` and
            carry no ``pr_number``; ``marker`` / ``llm`` / ``label`` may.
        pr_number: PR that introduced the change, when known. Enrichment only,
            never a reconciliation join key.
        similarity: Match score (0.0-1.0) for rename/move detection.
        relocated_to: For a ``relocated`` change, the new symbol id the symbol
            moved to (``change.id`` stays the OLD id so the ledger marks the
            origin entry relocated). Empty for all other change types.
    """

    change_type: str
    id: str
    surface: str = ""
    component: str = ""
    from_signature: str = ""
    to_signature: str = ""
    summary: str = ""
    confidence: str = "medium"
    source: str = "diff"
    pr_number: int | None = None
    similarity: float = 0.0
    relocated_to: str = ""

    def __post_init__(self) -> None:
        """Validate the diff vocabulary at its model boundary."""
        if self.change_type not in CHANGE_TYPES:
            raise ValueError(f"invalid change type: {self.change_type}")
        if self.surface and self.surface not in SURFACES:
            raise ValueError(f"invalid surface: {self.surface}")
        if self.source not in SOURCES:
            raise ValueError(f"invalid source: {self.source}")
        if not self.id:
            raise ValueError("change id must not be empty")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "change_type": self.change_type,
            "id": self.id,
            "surface": self.surface,
            "component": self.component,
            "from_signature": self.from_signature,
            "to_signature": self.to_signature,
            "summary": self.summary,
            "confidence": self.confidence,
            "source": self.source,
            "pr_number": self.pr_number,
            "similarity": self.similarity,
            "relocated_to": self.relocated_to,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SurfaceChange:
        """Deserialize from dictionary."""
        return cls(
            change_type=data["change_type"],
            id=data["id"],
            surface=data.get("surface", ""),
            component=data.get("component", ""),
            from_signature=data.get("from_signature", ""),
            to_signature=data.get("to_signature", ""),
            summary=data.get("summary", ""),
            confidence=data.get("confidence", "medium"),
            source=data.get("source", "diff"),
            pr_number=data.get("pr_number"),
            similarity=data.get("similarity", 0.0),
            relocated_to=data.get("relocated_to", ""),
        )


# =============================================================================
# LedgerEntry
# =============================================================================


@dataclass
class LedgerEntry:
    """One symbol's cumulative lifecycle record in the durable ledger.

    Attributes:
        id: Stable symbol id (the merge key, with ``release``).
        surface: One of ``SURFACES``.
        kind: Surface-specific element kind.
        component: Dynamo component area.
        status: One of ``LIFECYCLE_STATUSES``.
        stability: One of ``STABILITY_TIERS``.
        review_status: One of ``REVIEW_STATUSES``.
        confidence: ``high`` / ``medium`` / ``low``.
        source: One of ``SOURCES`` (the highest-authority source seen).
        added_in: Release the symbol first appeared in.
        deprecated_in: Release a deprecation was recorded.
        removal_target: Release removal is scheduled for, when known.
        removed_in: Release the symbol was removed in.
        relocated_to: Repository the crate/symbol moved to, when relocated.
        relocated_in: Release the relocation was recorded.
        last_seen: Most recent release the symbol was observed present.
        note: Free-text note (e.g. native marker note).
        migration_guidance: How users adapt (from LLM / strategy doc).
        pr_number: Associated PR, enrichment only.
        signature_history: List of ``{"release", "signature"}`` dicts.
    """

    id: str
    surface: str
    kind: str = ""
    component: str = ""
    status: str = "stable"
    stability: str = "stable"
    review_status: str = "auto"
    confidence: str = "medium"
    source: str = "diff"
    added_in: str = ""
    deprecated_in: str = ""
    removal_target: str = ""
    removed_in: str = ""
    relocated_to: str = ""
    relocated_in: str = ""
    last_seen: str = ""
    note: str = ""
    migration_guidance: str = ""
    pr_number: int | None = None
    signature_history: list[dict[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Reject invalid durable state before it reaches policy validation."""
        if not self.id:
            raise ValueError("ledger entry id must not be empty")
        if self.surface not in SURFACES:
            raise ValueError(f"invalid surface: {self.surface}")
        if self.status not in LIFECYCLE_STATUSES:
            raise ValueError(f"invalid lifecycle status: {self.status}")
        if self.stability not in STABILITY_TIERS:
            raise ValueError(f"invalid stability tier: {self.stability}")
        if self.review_status not in REVIEW_STATUSES:
            raise ValueError(f"invalid review status: {self.review_status}")
        if self.source not in SOURCES:
            raise ValueError(f"invalid source: {self.source}")

    @property
    def is_deprecated(self) -> bool:
        """True when the symbol is deprecated or scheduled for removal."""
        return self.status in {"deprecated", "scheduled_for_removal"}

    @property
    def needs_review(self) -> bool:
        """True when the entry awaits human confirmation."""
        return self.review_status == "needs_review"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary, including computed properties."""
        return {
            "id": self.id,
            "surface": self.surface,
            "kind": self.kind,
            "component": self.component,
            "status": self.status,
            "stability": self.stability,
            "review_status": self.review_status,
            "confidence": self.confidence,
            "source": self.source,
            "added_in": self.added_in,
            "deprecated_in": self.deprecated_in,
            "removal_target": self.removal_target,
            "removed_in": self.removed_in,
            "relocated_to": self.relocated_to,
            "relocated_in": self.relocated_in,
            "last_seen": self.last_seen,
            "note": self.note,
            "migration_guidance": self.migration_guidance,
            "pr_number": self.pr_number,
            "signature_history": [dict(h) for h in self.signature_history],
            "is_deprecated": self.is_deprecated,
            "needs_review": self.needs_review,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LedgerEntry:
        """Deserialize from dictionary (computed properties are ignored)."""
        return cls(
            id=data["id"],
            surface=data["surface"],
            kind=data.get("kind", ""),
            component=data.get("component", ""),
            status=data.get("status", "stable"),
            stability=data.get("stability", "stable"),
            review_status=data.get("review_status", "auto"),
            confidence=data.get("confidence", "medium"),
            source=data.get("source", "diff"),
            added_in=data.get("added_in", ""),
            deprecated_in=data.get("deprecated_in", ""),
            removal_target=data.get("removal_target", ""),
            removed_in=data.get("removed_in", ""),
            relocated_to=data.get("relocated_to", ""),
            relocated_in=data.get("relocated_in", ""),
            last_seen=data.get("last_seen", ""),
            note=data.get("note", ""),
            migration_guidance=data.get("migration_guidance", ""),
            pr_number=data.get("pr_number"),
            signature_history=[dict(h) for h in data.get("signature_history", [])],
        )


# =============================================================================
# SurfaceLedger
# =============================================================================


@dataclass
class SurfaceLedger:
    """The durable cross-release ledger envelope.

    Attributes:
        schema_version: ``LEDGER_SCHEMA_VERSION`` at write time.
        generated_at: ISO datetime string of last write.
        entries: Lifecycle records. Persisted sorted by ``id``.
    """

    schema_version: int = LEDGER_SCHEMA_VERSION
    generated_at: str = ""
    entries: list[LedgerEntry] = field(default_factory=list)

    def sorted_entries(self) -> list[LedgerEntry]:
        """Return entries ordered by ``id`` for deterministic output."""
        return sorted(self.entries, key=lambda e: e.id)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary with entries sorted by id."""
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "entries": [e.to_dict() for e in self.sorted_entries()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SurfaceLedger:
        """Deserialize from dictionary."""
        return cls(
            schema_version=data.get("schema_version", LEDGER_SCHEMA_VERSION),
            generated_at=data.get("generated_at", ""),
            entries=[LedgerEntry.from_dict(e) for e in data.get("entries", [])],
        )


# =============================================================================
# Extractor protocol (frozen for Phase B subagents)
# =============================================================================


@runtime_checkable
class SurfaceExtractor(Protocol):
    """Contract every Phase B extractor module satisfies.

    Each extractor module exposes a module-level ``SURFACE`` string (one of
    ``SURFACES``) and an ``extract`` callable matching this signature. The
    returned :class:`OperationResult` carries:

    - ``data["symbols"]``: ``list[dict]`` of ``SurfaceSymbol.to_dict()`` output
    - ``metadata["surface"]``: the surface string
    - ``metadata["covered"]``: ``bool`` -- ``False`` when the surface could not
      be extracted at this ref (so the diff engine records a coverage gap and
      never reports false removals)

    Failures append :class:`OpError` to ``errors`` rather than raising.
    """

    def __call__(self, repo_path: Path, release: str) -> OperationResult:
        """Extract this surface from ``repo_path`` at the given ``release``."""
        ...
