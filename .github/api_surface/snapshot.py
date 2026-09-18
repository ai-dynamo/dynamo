# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Snapshot orchestration + persistence for the API surface tracker.

``build_snapshot`` runs every surface extractor against one checked-out ref and
assembles a single :class:`SurfaceSnapshot` (symbols + a per-surface coverage
map). ``gather_changes`` combines the snapshot-to-snapshot diff with the native
deprecation-marker scan into one change list ready for the ledger merge.

Like the rest of the engine this layer is pure and offline: an audit workflow
can call ``build_snapshot`` on a worktree at the merge commit and
``gather_changes`` against the merge-base snapshot with no release-layout
assumptions. The ``default_*_path`` helpers are opt-in conveniences for the CLI
and backfill; nothing here writes unless asked.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from api_surface.annotations import apply_annotations, default_annotations_path
from api_surface.diff import diff_snapshots
from api_surface.extractors import (
    crd,
    helm,
    http,
    metrics,
    python_components,
    python_pyi,
    rust,
)
from api_surface.markers import scan_markers
from api_surface.models import (
    SNAPSHOT_SCHEMA_VERSION,
    SurfaceChange,
    SurfaceSnapshot,
    SurfaceSymbol,
)
from api_surface.results import OperationResult, OpError
from api_surface.stability import normalize_stability

# Single-arg ``extract(repo_path, release)`` extractors. ``python_pyi`` is
# called separately so its optional ``stubs`` override can be threaded through.
_SURFACE_EXTRACTORS = (
    crd.extract,
    helm.extract,
    http.extract,
    metrics.extract,
    rust.extract,
)


def _now() -> str:
    """Current UTC timestamp as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _merge_coverage(coverage: dict[str, bool], result: OperationResult) -> None:
    """Fold one extractor's coverage into the running snapshot coverage map.

    A multi-surface extractor (``http`` owns http/env/config) reports a
    ``metadata['coverage']`` dict; single-surface extractors report
    ``metadata['surface']`` + ``metadata['covered']``.
    """
    cov = result.metadata.get("coverage")
    if isinstance(cov, dict):
        for surface, ok in cov.items():
            coverage[str(surface)] = bool(ok)
        return
    surface = result.metadata.get("surface")
    if surface:
        coverage[str(surface)] = bool(result.metadata.get("covered"))


def _deduplicate_symbols(
    symbols: list[SurfaceSymbol],
) -> tuple[list[SurfaceSymbol], list[OpError]]:
    """Drop byte-identical repeats and reject conflicting stable IDs."""
    by_id: dict[str, SurfaceSymbol] = {}
    errors: list[OpError] = []
    for symbol in symbols:
        existing = by_id.get(symbol.id)
        if existing is None:
            by_id[symbol.id] = symbol
            continue
        if existing.to_dict() == symbol.to_dict():
            continue
        errors.append(
            OpError(
                operation="build_snapshot",
                message=f"conflicting symbols share stable id: {symbol.id}",
                details={"first_kind": existing.kind, "second_kind": symbol.kind},
            )
        )
    return sorted(by_id.values(), key=lambda symbol: symbol.id), errors


def build_snapshot(
    repo_path: str | Path,
    release: str,
    ref: str = "",
    repo: str = "ai-dynamo/dynamo",
    stubs: list[tuple[str, str]] | None = None,
) -> OperationResult:
    """Run every surface extractor at ``repo_path`` into one snapshot.

    Args:
        repo_path: Root of a checked-out source tree.
        release: Release version string (3-dotted, e.g. ``1.2.0``).
        ref: Git ref the tree is at; defaults to ``v<release>``.
        repo: Source repository identifier recorded on the snapshot.
        stubs: Optional ``(rel_path, module)`` overrides for the python
            extractor (tests point this at synthetic fixtures).

    Returns:
        :class:`OperationResult` with ``data['snapshot']`` =
        :meth:`SurfaceSnapshot.to_dict`. Extractor failures are surfaced as
        ``errors`` (each tagged with its surface) but never abort the build:
        the uncovered surface is simply marked ``False`` in the coverage map so
        the diff engine never reads it as a removal.
    """
    repo_path = Path(repo_path)
    symbols: list[SurfaceSymbol] = []
    coverage: dict[str, bool] = {}
    coverage_detail: dict[str, bool] = {}
    errors = []

    results = [fn(repo_path, release) for fn in _SURFACE_EXTRACTORS]
    results.append(python_pyi.extract(repo_path, release, stubs=stubs))
    # Real ``components/`` .py public API on the same "python" surface. Reports
    # only ``coverage_detail['python:components']`` so it supplements (never
    # clobbers) the stub extractor's top-level ``python`` coverage bit.
    results.append(python_components.extract(repo_path, release))
    for result in results:
        symbols.extend(
            SurfaceSymbol.from_dict(d) for d in result.data.get("symbols", [])
        )
        _merge_coverage(coverage, result)
        detail = result.metadata.get("coverage_detail")
        if isinstance(detail, dict):
            for key, ok in detail.items():
                coverage_detail[str(key)] = bool(ok)
        errors.extend(result.errors)

    symbols, duplicate_errors = _deduplicate_symbols(symbols)
    errors.extend(duplicate_errors)

    # Infer first, then let an explicit project annotation override the tier.
    for sym in symbols:
        normalize_stability(sym)
    annotation_result = apply_annotations(symbols, default_annotations_path(repo_path))
    errors.extend(annotation_result.errors)

    snapshot = SurfaceSnapshot(
        release=release,
        ref=ref or f"v{release}",
        repo=repo,
        generated_at=_now(),
        schema_version=SNAPSHOT_SCHEMA_VERSION,
        coverage=coverage,
        coverage_detail=coverage_detail,
        symbols=symbols,
    )
    return OperationResult(
        data={"snapshot": snapshot.to_dict()},
        errors=errors,
        metadata={
            "release": release,
            "ref": snapshot.ref,
            "symbol_count": len(symbols),
            "covered_surfaces": sorted(s for s, ok in coverage.items() if ok),
            "coverage_gaps": sorted(s for s, ok in coverage.items() if not ok),
            "annotations_applied": annotation_result.metadata.get("applied", 0),
        },
    )


def gather_changes(
    old: SurfaceSnapshot,
    new: SurfaceSnapshot,
    repo_path: str | Path | None = None,
    release: str = "",
    include_markers: bool = True,
) -> OperationResult:
    """Combine the diff between two snapshots with the native-marker scan.

    The diff supplies add/remove/signature/rename events (source ``diff``); the
    marker scan supplies high-confidence ``deprecated_api`` events (source
    ``marker``) read straight from ``#[deprecated]`` / ``@deprecated`` in the
    new tree. Both feed :func:`ledger.merge_changes` unchanged.

    Args:
        old: The prior release's snapshot.
        new: The current release's snapshot.
        repo_path: Checked-out tree for the marker scan; when ``None`` markers
            are skipped (diff-only).
        release: Release the markers belong to; defaults to ``new.release``.
        include_markers: Set ``False`` to diff only.

    Returns:
        :class:`OperationResult` with ``data['changes']`` =
        ``list[SurfaceChange.to_dict()]`` sorted deterministically, plus
        ``data['coverage_gaps']`` carried from the diff.
    """
    diff_result = diff_snapshots(old, new)
    changes = [SurfaceChange.from_dict(c) for c in diff_result.data.get("changes", [])]
    errors = list(diff_result.errors)

    if include_markers and repo_path is not None:
        marker_result = scan_markers(Path(repo_path), release or new.release)
        changes.extend(
            SurfaceChange.from_dict(c) for c in marker_result.data.get("changes", [])
        )
        errors.extend(marker_result.errors)

    changes.sort(key=lambda c: (c.surface, c.id, c.change_type, c.source))
    return OperationResult(
        data={
            "changes": [c.to_dict() for c in changes],
            "coverage_gaps": diff_result.data.get("coverage_gaps", []),
        },
        errors=errors,
        metadata={
            "change_count": len(changes),
            "diff_count": len(diff_result.data.get("changes", [])),
            "marker_count": len(changes) - len(diff_result.data.get("changes", [])),
        },
    )


def load_snapshot(path: str | Path) -> SurfaceSnapshot:
    """Load a snapshot from JSON."""
    return SurfaceSnapshot.from_dict(json.loads(Path(path).read_text()))


def save_snapshot(snapshot: SurfaceSnapshot, path: str | Path) -> None:
    """Persist a snapshot as sorted, pretty JSON (atomic write)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(snapshot.to_dict(), indent=2, sort_keys=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload + "\n")
    tmp.replace(path)


def default_snapshot_path(release: str, root: str | Path = ".") -> Path:
    """Return the project-local path for one immutable release snapshot."""
    return Path(root) / ".github" / "api-surface" / "snapshots" / f"{release}.json"


def default_ledger_path(root: str | Path = ".") -> Path:
    """Return the project-local durable ledger path."""
    return Path(root) / ".github" / "api-surface" / "ledger.json"


def default_provenance_path(root: str | Path = ".") -> Path:
    """Repo-standard crate-provenance path used for relocation detection."""
    return Path(root) / ".github" / "api-surface" / "crate-provenance.yaml"
