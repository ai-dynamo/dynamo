# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Broken-promise validation over the durable ledger.

The stability contract the tracker enforces: a ``stable`` symbol must be
deprecated for at least one release before it is removed, and must not be
removed before any ``removal_target`` the ledger promised. Removing a
``preview`` / ``experimental`` symbol is allowed and reported as informational,
never a violation -- that is the whole point of the tiers.

``validate_ledger`` is pure: it reads the ledger + a :class:`SuppressionSet` and
returns the violations. The CLI maps a non-empty violation list to a non-zero
exit so this can gate a release pipeline. Waivers in ``suppressions.yaml``
silence individual findings without weakening the check globally.
"""

from __future__ import annotations

from api_surface.models import LedgerEntry, SurfaceLedger
from api_surface.results import OperationResult
from api_surface.suppressions import SuppressionSet, version_key

# Stability tiers whose removal breaks the public-API promise. Non-stable tiers
# are exempt by design.
_PROTECTED_TIERS = {"stable"}

# Contract surfaces need the longer policy window; every other stable surface
# must survive at least one intervening release.
_MINIMUM_WINDOWS = {"http": 3, "crd": 3}


def _violation(entry: LedgerEntry, kind: str, detail: str) -> dict[str, object]:
    """Build one violation record from a ledger entry."""
    return {
        "id": entry.id,
        "surface": entry.surface,
        "component": entry.component,
        "stability": entry.stability,
        "kind": kind,
        "detail": detail,
        "deprecated_in": entry.deprecated_in,
        "removed_in": entry.removed_in,
        "removal_target": entry.removal_target,
    }


def _release_distance(start: str, end: str) -> int:
    """Return a conservative release count between two semantic versions."""
    start_key = version_key(start)
    end_key = version_key(end)
    if len(start_key) < 2 or len(end_key) < 2 or end_key <= start_key:
        return 0
    if end_key[0] != start_key[0]:
        # A major boundary satisfies the longest OSS window defined here; the
        # exact number of intervening minor releases is immaterial to the gate.
        return max(_MINIMUM_WINDOWS.values())
    if end_key[1] != start_key[1]:
        return end_key[1] - start_key[1]
    start_patch = start_key[2] if len(start_key) > 2 else 0
    end_patch = end_key[2] if len(end_key) > 2 else 0
    return end_patch - start_patch


def _check_entry(entry: LedgerEntry) -> dict[str, object] | None:
    """Return a violation for a protected-tier entry, or ``None`` if compliant."""
    if entry.stability not in _PROTECTED_TIERS:
        return None

    if entry.status == "removed" and not entry.deprecated_in:
        return _violation(
            entry,
            "removed_without_deprecation",
            f"stable symbol removed in {entry.removed_in or '?'} with no prior deprecation",
        )

    if (
        entry.status in {"deprecated", "scheduled_for_removal"}
        and not entry.removal_target
    ):
        return _violation(
            entry,
            "missing_removal_target",
            f"stable symbol deprecated in {entry.deprecated_in or '?'} with no removal target",
        )

    if entry.status in {"deprecated", "scheduled_for_removal"} and not (
        entry.note or entry.migration_guidance
    ):
        return _violation(
            entry,
            "missing_migration_guidance",
            "stable deprecation has no successor or migration guidance",
        )

    window_end = entry.removal_target or entry.removed_in
    if entry.deprecated_in and window_end:
        required = _MINIMUM_WINDOWS.get(entry.surface, 1)
        actual = _release_distance(entry.deprecated_in, window_end)
        if actual < required:
            return _violation(
                entry,
                "minimum_window_not_met",
                f"{entry.surface} deprecation in {entry.deprecated_in} targets "
                f"{window_end}, only {actual} release(s) later; {required} required",
            )

    if entry.status != "removed":
        return None

    if (
        entry.removal_target
        and entry.removed_in
        and version_key(entry.removed_in) < version_key(entry.removal_target)
    ):
        return _violation(
            entry,
            "premature_removal",
            f"removed in {entry.removed_in} before promised removal target {entry.removal_target}",
        )

    return None


def validate_ledger(
    ledger: SurfaceLedger, suppressions: SuppressionSet | None = None
) -> OperationResult:
    """Validate the ledger's stable-surface removal promises.

    Args:
        ledger: The durable ledger to check.
        suppressions: Waivers that silence individual findings; ``None`` means
            no waivers.

    Returns:
        :class:`OperationResult` with ``data['violations']`` (list of records),
        ``data['waived']`` (count of findings silenced by suppressions), and
        ``metadata`` summarizing counts by ``kind``. ``errors`` stays empty --
        a broken promise is data the caller acts on, not an operation failure.
    """
    suppressions = suppressions or SuppressionSet()
    violations: list[dict[str, object]] = []
    waived = 0
    releases = [
        release
        for entry in ledger.entries
        for release in (
            entry.added_in,
            entry.deprecated_in,
            entry.removal_target,
            entry.removed_in,
            entry.relocated_in,
            entry.last_seen,
        )
        if release
    ]
    current_release = max(releases, key=version_key, default="")

    for entry in ledger.sorted_entries():
        finding = _check_entry(entry)
        if finding is None:
            continue
        if suppressions.is_suppressed(
            entry.id, surface=entry.surface, release=current_release
        ):
            waived += 1
            continue
        violations.append(finding)

    by_kind: dict[str, int] = {}
    for v in violations:
        kind = str(v["kind"])
        by_kind[kind] = by_kind.get(kind, 0) + 1

    return OperationResult(
        data={"violations": violations, "waived": waived},
        metadata={
            "violation_count": len(violations),
            "waived_count": waived,
            "by_kind": by_kind,
        },
    )
