# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Durable cross-release ledger merge + persistence.

``merge_changes`` is a pure function: given the current ledger, the new
snapshot, and the diff/signal changes for a release, it returns a brand-new
:class:`SurfaceLedger`. It is idempotent keyed by ``(id, release)`` -- merging
the same release twice yields the same ledger, so re-running across nightly RCs
is safe. Presence is driven by the snapshot (gated by its coverage map), never
by the absence of a change event, so coverage gaps never fabricate removals.

Persistence (``load_ledger`` / ``save_ledger``) is split out and the ledger
file is git-tracked, which is the durability backstop while keeping the engine
usable by workflow consumers.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from api_surface.models import (
    LEDGER_SCHEMA_VERSION,
    LedgerEntry,
    SurfaceChange,
    SurfaceLedger,
    SurfaceSnapshot,
)

# Source authority, highest wins. Native markers are the primary deprecation
# signal; contributor labels are the weakest.
_SOURCE_RANK = {"marker": 3, "diff": 2, "llm": 1, "label": 0}

# Sources whose findings must be human-confirmed before they are authoritative.
LOW_CONFIDENCE_SOURCES = {"llm", "label"}

# Surfaces extracted by regex / line-scan (no parser). Their *removals* are
# candidate breaks needing human confirmation, not asserted ones: a struct that
# moved or was reclassified reads as a disappearance on these surfaces (the
# ``ApiError`` class of false positive). Removals here are gated ``needs_review``
# so a consumer never presents them as a confident break.
HEURISTIC_SURFACES = {"config", "env", "helm"}


def _now() -> str:
    """Current UTC timestamp as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def stronger_source(current: str, incoming: str) -> str:
    """Return the higher-authority of two sources (marker > diff > llm > label)."""
    if _SOURCE_RANK.get(incoming, -1) > _SOURCE_RANK.get(current, -1):
        return incoming
    return current


# Backwards-compatible private alias for in-module callers.
_stronger_source = stronger_source


def gate_review(entry: LedgerEntry) -> None:
    """Park an entry ``needs_review`` when its strongest signal is not trustworthy.

    The single home for the review-gating policy. Two triggers park an entry for
    human confirmation: a low-confidence *source* (llm / label), or a *removal on
    a heuristic surface* (``HEURISTIC_SURFACES`` -- regex / line-scan extractions
    whose disappearances are candidate breaks, not asserted ones). A ``confirmed``
    entry is never demoted. An entry meeting neither trigger is (re)set ``auto``
    so the policy is idempotent under re-merge and a reappeared symbol clears the
    review flag its earlier removal raised.
    """
    if entry.review_status == "confirmed":
        return
    low_confidence_source = entry.source in LOW_CONFIDENCE_SOURCES
    heuristic_removal = (
        entry.status == "removed" and entry.surface in HEURISTIC_SURFACES
    )
    entry.review_status = (
        "needs_review" if (low_confidence_source or heuristic_removal) else "auto"
    )


def apply_signal(
    entry: LedgerEntry,
    *,
    source: str,
    release: str,
    pr_number: int | None = None,
    component: str = "",
    deprecated: bool = False,
    note: str = "",
    migration_guidance: str = "",
) -> None:
    """Fold one low-confidence (llm / label) signal into a ledger entry in place.

    Shared by the LLM backfill and the contributor-label ingest, which both
    reconcile PR-keyed prose onto an already-matched symbol entry. Gap-filling
    and idempotent: PR/component/guidance fill only when empty, ``source`` only
    ever strengthens, a deprecation never overrides an already removed/relocated
    entry, and the entry is gated ``needs_review`` per :func:`gate_review`.
    """
    entry.migration_guidance = entry.migration_guidance or migration_guidance
    if entry.pr_number is None:
        entry.pr_number = pr_number
    entry.component = entry.component or component
    entry.source = stronger_source(entry.source, source)
    if deprecated and entry.status not in {"removed", "relocated"}:
        entry.status = "deprecated"
        entry.deprecated_in = entry.deprecated_in or release
        entry.note = entry.note or note
    gate_review(entry)


def _relocation_repo(entry: LedgerEntry, provenance: dict[str, dict]) -> str:
    """Return the new home repo if this entry's crate left the monorepo."""
    if entry.surface != "rust" or ":" not in entry.id:
        return ""
    crate = entry.id.split(":", 1)[1].split("::", 1)[0]
    info = provenance.get(crate, {})
    if info.get("status") == "external":
        return str(info.get("repo", ""))
    return ""


def _record_signature(entry: LedgerEntry, release: str, signature: str) -> None:
    """Append a signature observation, de-duplicated by (release, signature)."""
    if not signature:
        return
    pair = {"release": release, "signature": signature}
    if pair not in entry.signature_history:
        entry.signature_history.append(pair)


def _upsert_present(
    index: dict[str, LedgerEntry], snapshot: SurfaceSnapshot, release: str
) -> None:
    """Create/update an entry for every symbol present in the snapshot."""
    for sym in snapshot.symbols:
        entry = index.get(sym.id)
        if entry is None:
            entry = LedgerEntry(
                id=sym.id, surface=sym.surface, added_in=release, status="added"
            )
            index[sym.id] = entry
        entry.surface = sym.surface
        entry.kind = sym.kind or entry.kind
        entry.component = sym.component or entry.component
        entry.stability = sym.stability or entry.stability
        entry.last_seen = release
        if not entry.added_in:
            entry.added_in = release
        if entry.status in {"removed", "relocated"}:
            entry.status = "added"
            entry.added_in = release
            entry.removed_in = ""
            entry.relocated_to = ""
            entry.relocated_in = ""
        elif entry.status == "added" and entry.added_in != release:
            entry.status = "stable"
        if sym.deprecated:
            entry.status = "deprecated"
            if not entry.deprecated_in:
                entry.deprecated_in = release
            entry.note = sym.deprecated_note or entry.note
            entry.removal_target = (
                str(sym.metadata.get("removal_target", "")) or entry.removal_target
            )
            entry.source = _stronger_source(entry.source, "marker")


def _reconcile_removals(
    index: dict[str, LedgerEntry],
    snapshot: SurfaceSnapshot,
    release: str,
    provenance: dict,
) -> None:
    """Mark entries absent from a covered surface as removed or relocated."""
    covered = {s for s, ok in snapshot.coverage.items() if ok}
    present = {sym.id for sym in snapshot.symbols}
    for entry in index.values():
        if entry.surface not in covered or entry.id in present:
            continue
        if entry.status in {"removed", "relocated"}:
            continue
        repo = _relocation_repo(entry, provenance)
        if repo:
            entry.status = "relocated"
            entry.relocated_to = repo
            if not entry.relocated_in:
                entry.relocated_in = release
        else:
            entry.status = "removed"
            if not entry.removed_in:
                entry.removed_in = release


def _apply_deprecation(entry: LedgerEntry, change: SurfaceChange, release: str) -> None:
    """Record a deprecation from a marker / LLM / label signal."""
    entry.status = "deprecated"
    if not entry.deprecated_in:
        entry.deprecated_in = release
    if change.summary:
        entry.note = change.summary
    entry.confidence = change.confidence or entry.confidence


def _rust_crate_of(sid: str) -> str:
    """Crate from a ``rust:<crate>::...`` id (or ``""`` when not parseable)."""
    if not sid.startswith("rust:"):
        return ""
    return sid[len("rust:") :].split("::", 1)[0]


def _reconcile_rust_marker(index: dict[str, LedgerEntry], marker_id: str) -> str | None:
    """Map a marker rust id to an existing entry by ``(crate, trailing name)``.

    The marker scanner derives only the file-level module path, so a marker on
    an item inside an inline ``mod`` or an ``impl`` block lands a shorter id
    than the extractor's fully-qualified one. When the marker id has no exact
    entry, match it to the unique existing rust entry sharing its crate and
    trailing item name; ambiguous (>1) or absent matches return ``None`` and
    the marker keeps its own entry.
    """
    crate = _rust_crate_of(marker_id)
    trailing = marker_id.rsplit("::", 1)[-1]
    matches = [
        e.id
        for e in index.values()
        if e.surface == "rust"
        and _rust_crate_of(e.id) == crate
        and e.id.rsplit("::", 1)[-1] == trailing
    ]
    return matches[0] if len(matches) == 1 else None


def _apply_change(
    index: dict[str, LedgerEntry], change: SurfaceChange, release: str
) -> None:
    """Fold one non-presence change (signature/rename/deprecation/relocation)."""
    target_id = change.id
    if (
        change.source == "marker"
        and change.surface == "rust"
        and target_id not in index
    ):
        redirected = _reconcile_rust_marker(index, target_id)
        if redirected is not None:
            target_id = redirected

    entry = index.get(target_id)
    if entry is None:
        # Seed source from the creating change so a label/llm-only entry carries
        # honest provenance (not the default ``diff``) -- gate_review then keys on
        # the entry's real strongest source.
        entry = LedgerEntry(
            id=target_id,
            surface=change.surface,
            last_seen=release,
            source=change.source,
        )
        index[target_id] = entry
    entry.component = change.component or entry.component
    entry.pr_number = (
        change.pr_number if change.pr_number is not None else entry.pr_number
    )
    entry.source = _stronger_source(entry.source, change.source)

    if change.change_type in {"signature_changed", "renamed_function"}:
        _record_signature(entry, release, change.to_signature)
    if change.change_type in {"deprecated_api"}:
        _apply_deprecation(entry, change, release)
    if change.change_type in {"relocated"}:
        # A symbol-level relocation overrides any transient ``removed`` that the
        # snapshot-absence reconcile set earlier in this same merge: the symbol
        # moved, it was not removed.
        entry.status = "relocated"
        entry.removed_in = ""
        if not entry.relocated_in:
            entry.relocated_in = release
        if change.relocated_to:
            entry.relocated_to = change.relocated_to


def merge_changes(
    ledger: SurfaceLedger,
    changes: list[SurfaceChange],
    release: str,
    snapshot: SurfaceSnapshot | None = None,
    provenance: dict[str, dict] | None = None,
) -> SurfaceLedger:
    """Merge a release's snapshot + changes into a new ledger (idempotent).

    Args:
        ledger: The current ledger (not mutated).
        changes: Diff/signal changes for this release.
        release: Release version string the changes belong to.
        snapshot: The release's full snapshot. When provided, presence drives
            add/stable/removed transitions (gated by ``snapshot.coverage``).
            When ``None`` (e.g. a mid-release label ingest), only ``changes``
            are folded in.
        provenance: Crate-to-home-repo map; drives relocated-vs-removed.

    Returns:
        A new :class:`SurfaceLedger` with merged entries, sorted by id.
    """
    provenance = provenance or {}
    index: dict[str, LedgerEntry] = {e.id: copy.deepcopy(e) for e in ledger.entries}

    if snapshot is not None:
        _upsert_present(index, snapshot, release)
        _reconcile_removals(index, snapshot, release, provenance)

    for change in changes:
        _apply_change(index, change, release)

    # Re-gate every entry against the final merged state. Presence-driven
    # removals never pass through _apply_change, so this single sweep is where
    # heuristic-surface removals get parked needs_review (and reappeared symbols
    # get cleared back to auto).
    for entry in index.values():
        gate_review(entry)

    return SurfaceLedger(generated_at=_now(), entries=list(index.values()))


def deprecations(
    ledger: SurfaceLedger, include_unconfirmed: bool = False
) -> list[LedgerEntry]:
    """Return the ledger's currently-deprecated entries, newest deprecation first.

    This is the importable seam release tooling reads instead of re-running an
    LLM scan: the ledger is the authoritative deprecation record. By default
    only auto/confirmed entries are returned; pass
    ``include_unconfirmed=True`` to also surface ``needs_review`` ones (e.g. for
    a human triage queue).

    Args:
        ledger: The durable ledger to read.
        include_unconfirmed: Include entries still awaiting review.

    Returns:
        :class:`LedgerEntry` objects that are deprecated or scheduled for
        removal, sorted by ``deprecated_in`` descending then ``id``.
    """
    out = [
        e
        for e in ledger.entries
        if e.is_deprecated and (include_unconfirmed or not e.needs_review)
    ]
    out.sort(key=lambda e: (e.deprecated_in, e.id), reverse=True)
    return out


def pending_review(ledger: SurfaceLedger) -> list[LedgerEntry]:
    """Return entries awaiting human confirmation, sorted by id.

    These are the low-confidence (llm / label) findings the merge parked as
    ``needs_review``; ``deprecations()`` hides them until a human confirms via
    :func:`confirm_review`.
    """
    return sorted((e for e in ledger.entries if e.needs_review), key=lambda e: e.id)


def confirm_review(ledger: SurfaceLedger, ids: list[str]) -> int:
    """Promote the given ``needs_review`` entries to ``confirmed`` (in place).

    Only entries currently ``needs_review`` are touched; ids that are unknown or
    already ``auto`` / ``confirmed`` are ignored. Returns the number confirmed so
    the CLI can report how many of the requested ids actually moved.
    """
    wanted = set(ids)
    confirmed = 0
    for entry in ledger.entries:
        if entry.id in wanted and entry.review_status == "needs_review":
            entry.review_status = "confirmed"
            confirmed += 1
    return confirmed


# Ledger schema migrations: ``version -> fn`` that upgrades a raw dict from that
# version to the next. Empty today (v1 is current); the chain below applies them
# in order so a bump only needs one new entry, never a rewrite of load_ledger.
_LEDGER_MIGRATIONS: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {}


def migrate_ledger_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Upgrade a raw ledger dict to :data:`LEDGER_SCHEMA_VERSION` in place.

    Walks the registered migration chain from the file's recorded version up to
    the current one. A pre-versioning file (no ``schema_version``) is treated as
    v1. Unknown old or future schemas fail closed rather than being interpreted
    with possibly incompatible semantics.
    """
    version = int(data.get("schema_version", LEDGER_SCHEMA_VERSION))
    if version > LEDGER_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported ledger schema {version}; expected {LEDGER_SCHEMA_VERSION}"
        )
    while version < LEDGER_SCHEMA_VERSION:
        if version not in _LEDGER_MIGRATIONS:
            raise ValueError(f"unsupported ledger schema {version}")
        data = _LEDGER_MIGRATIONS[version](data)
        version += 1
    data["schema_version"] = version
    return data


def load_provenance(path: str | Path) -> dict[str, dict]:
    """Load the crate-to-home-repo map from ``crate_provenance.yaml``.

    Returns the inner ``crates`` mapping (crate name -> ``{repo, status, ...}``),
    or an empty dict when the file is absent.
    """
    path = Path(path)
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    crates = data.get("crates") or {}
    return {str(name): dict(info) for name, info in crates.items()}


def load_ledger(path: str | Path) -> SurfaceLedger:
    """Load a ledger from JSON, returning an empty ledger if absent.

    The raw dict is run through :func:`migrate_ledger_dict` first so an older
    on-disk schema is upgraded before deserialization.
    """
    path = Path(path)
    if not path.exists():
        return SurfaceLedger()
    return SurfaceLedger.from_dict(migrate_ledger_dict(json.loads(path.read_text())))


def save_ledger(ledger: SurfaceLedger, path: str | Path) -> None:
    """Persist a ledger as sorted, pretty JSON (atomic write)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(ledger.to_dict(), indent=2, sort_keys=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload + "\n")
    tmp.replace(path)
