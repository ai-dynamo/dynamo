# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Code-grounded surface diff engine.

Compares two :class:`SurfaceSnapshot` objects and emits :class:`SurfaceChange`
events (added / removed / signature_changed / renamed_function / relocated).
Pure and deterministic: output is sorted, signatures are canonicalized before
comparison, and a surface that is not covered in *both* snapshots is skipped
entirely so a coverage gap is never misread as a wave of removals. Rust is
additionally gated per-crate (``coverage_detail``) and rust crate moves are
paired as ``relocated`` rather than removal+addition, as is a whole rust crate
cleanly folded into another under a preserved module prefix (``CRATE_MERGES``).
On the structured non-rust surfaces (python / crd / config), a renamed
container (a class / namespace whose members move in lockstep, e.g.
``backend.EngineConfig`` -> ``backend.LlmRegistration``) is likewise paired
member-by-member as ``relocated`` rather than a wall of removals plus additions.
"""

from __future__ import annotations

import time
from difflib import SequenceMatcher

from api_surface.models import SurfaceChange, SurfaceSnapshot, SurfaceSymbol
from api_surface.results import OperationResult

# Minimum signature similarity for a removed+added pair to be reported as a
# rename rather than two independent events.
RENAME_SIMILARITY_THRESHOLD = 0.85

# A container (class / struct / namespace) rename is recognized when a removed
# container and an added sibling container share at least this many member
# leaves and this fraction of the smaller member set. The dual guard stops a
# coincidental pair of sibling classes that merely share a couple of generic
# member names (``__init__``, ``to_dict``) from fabricating a rename.
CONTAINER_RENAME_MIN_SHARED = 3
CONTAINER_RENAME_OVERLAP = 0.5

# Rust crates folded *wholesale and cleanly* into another crate under a single
# module prefix: ``{old_crate: (new_crate, module_prefix)}``. A symbol
# ``rust:<old>::<rest>`` must reappear *verbatim* as
# ``rust:<new>::<prefix>::<rest>``; the pair is then a relocation, not an
# independent removal + addition. The per-crate coverage gate hides the old
# crate (absent at the new ref), so :func:`_detect_crate_merges` reads the
# *ungated* symbol sets to recover the move.
#
# This is intentionally strict: it only fires when the inner module path is
# preserved within the *same* repo. It is deliberately empty today.
# ``dynamo-protocols`` / ``dynamo-parsers`` / ``dynamo-tokenizers`` are NOT
# listed: they were extracted out of the monorepo into the published
# ``ai-dynamo/frontend-crates`` repo (consumed back as crates.io deps), so their
# source is simply absent at ``main`` -- a cross-repo move no in-repo diff can
# pair. The conservative gate hides the now-absent crate (no false removals);
# tracking that surface requires scanning frontend-crates (see the quality-audit
# doc, "Cross-Repo Extraction").
# Example shape for a real in-repo clean merge:
#     "dynamo-oldcrate": ("dynamo-newcrate", "oldcrate"),
CRATE_MERGES: dict[str, tuple[str, str]] = {}


def canonicalize_signature(signature: str) -> str:
    """Normalize a signature so cosmetic churn does not register as a change.

    Collapses all runs of whitespace to a single space and trims the ends.
    Extractors emit already-typed signatures; this guards against incidental
    formatting differences between releases.
    """
    return " ".join(signature.split())


def _covered_surfaces(old: SurfaceSnapshot, new: SurfaceSnapshot) -> set[str]:
    """Surfaces extracted in both snapshots (the only ones safe to diff)."""
    old_cov = {s for s, ok in old.coverage.items() if ok}
    new_cov = {s for s, ok in new.coverage.items() if ok}
    return old_cov & new_cov


def _covered_rust_crates(old: SurfaceSnapshot, new: SurfaceSnapshot) -> set[str]:
    """Rust crates covered in BOTH snapshots' ``coverage_detail``.

    Keys are ``rust:<crate>``; the returned set holds the bare ``<crate>``
    names. A curated crate absent or not-covered at either ref is excluded so
    the diff never reads a crate that simply moved or was renamed (e.g. the
    ``kvbm-*`` splits) as a wave of removals.
    """
    old_ok = {
        k.split(":", 1)[1] for k, v in old.coverage_detail.items() if v and ":" in k
    }
    new_ok = {
        k.split(":", 1)[1] for k, v in new.coverage_detail.items() if v and ":" in k
    }
    return old_ok & new_ok


def _rust_crate(sid: str) -> str:
    """Crate name from a ``rust:<crate>::...`` id (or ``""`` if not parseable)."""
    if not sid.startswith("rust:"):
        return ""
    return sid[len("rust:") :].split("::", 1)[0]


def _index(
    symbols: list[SurfaceSymbol], surfaces: set[str], rust_crates: set[str]
) -> dict[str, SurfaceSymbol]:
    """Index symbols by id, restricted to covered surfaces and (for rust) crates.

    Rust symbols are additionally gated per-crate: a rust symbol is indexed
    only when its crate is covered in both snapshots, so a per-crate coverage
    gap is skipped rather than misread as removals.
    """
    out: dict[str, SurfaceSymbol] = {}
    for s in symbols:
        if s.surface not in surfaces:
            continue
        if s.surface == "rust" and _rust_crate(s.id) not in rust_crates:
            continue
        out[s.id] = s
    return out


def _signature_change(old: SurfaceSymbol, new: SurfaceSymbol) -> SurfaceChange | None:
    """Emit a signature_changed event when canonical signatures differ."""
    before = canonicalize_signature(old.signature)
    after = canonicalize_signature(new.signature)
    if before == after:
        return None
    return SurfaceChange(
        change_type="signature_changed",
        id=new.id,
        surface=new.surface,
        component=new.component,
        from_signature=before,
        to_signature=after,
        summary=f"Signature changed for {new.id}",
        confidence="high",
        source="diff",
    )


def _split_container(sid: str) -> tuple[str | None, str]:
    """Split an id into ``(container, leaf)`` on the last ``::`` or ``.``.

    ``rust:crate::mod::Foo`` -> ``("rust:crate::mod", "Foo")`` and
    ``python:pkg.mod.Class.attr`` -> ``("python:pkg.mod.Class", "attr")``.
    Unstructured ids (http / env / metric, no separator) yield ``(None, sid)``.
    """
    p_colon = sid.rfind("::")
    p_dot = sid.rfind(".")
    if p_colon > p_dot:
        return sid[:p_colon], sid[p_colon + 2 :]
    if p_dot != -1:
        return sid[:p_dot], sid[p_dot + 1 :]
    return None, sid


def _rename_container(sid: str) -> str | None:
    """The id with its trailing item segment removed, or ``None`` if unstructured.

    A rename keeps the symbol's container (module / class / struct) and changes
    only the leaf name, so two ids may pair as a rename only when their
    containers match. The container is the id minus the segment after the last
    ``::`` (rust) or ``.`` (python / crd / config). Unstructured ids (http / env
    / metric, no separator) return ``None`` so the guard does not apply and
    prior behavior is preserved.
    """
    return _split_container(sid)[0]


def _best_rename_match(
    removed: SurfaceSymbol, candidates: dict[str, SurfaceSymbol]
) -> tuple[str, float]:
    """Return the best same-surface, same-container added candidate + similarity.

    Requires non-empty signatures on both sides: similarity cannot be
    established from empty strings (``SequenceMatcher`` scores two empty
    strings as a perfect match, which would fabricate renames). The
    same-container guard (when both ids are structured) stops fabricated rust
    pairings such as ``crateA::Foo::new`` <-> ``crateB::Bar::new`` whose
    identical ``pub fn new() -> Self`` signatures would otherwise match.
    """
    best_id, best_score = "", 0.0
    removed_sig = canonicalize_signature(removed.signature)
    if not removed_sig:
        return best_id, best_score
    removed_container = _rename_container(removed.id)
    for cand_id, cand in candidates.items():
        if cand.surface != removed.surface:
            continue
        cand_container = _rename_container(cand_id)
        if (
            removed_container is not None
            and cand_container is not None
            and cand_container != removed_container
        ):
            continue
        cand_sig = canonicalize_signature(cand.signature)
        if not cand_sig:
            continue
        score = SequenceMatcher(None, removed_sig, cand_sig).ratio()
        if score > best_score:
            best_id, best_score = cand_id, score
    return best_id, best_score


def _rust_suffix(sid: str) -> tuple[str, str]:
    """Split a rust id into ``(crate, rest)`` where rest is the id after the crate.

    ``rust:dynamo-llm::storage::Foo`` -> ``("dynamo-llm", "storage::Foo")``;
    ``rust:dynamo-llm::Foo`` -> ``("dynamo-llm", "Foo")``. A non-rust or
    crate-only id yields an empty ``rest``.
    """
    if not sid.startswith("rust:"):
        return "", ""
    crate, _, rest = sid[len("rust:") :].partition("::")
    return crate, rest


def _detect_relocations(
    removed: dict[str, SurfaceSymbol], added: dict[str, SurfaceSymbol]
) -> list[SurfaceChange]:
    """Pair rust symbols that moved crate but kept their qualified path.

    A removed and an added rust id sharing the same crate-relative ``rest`` but
    a different crate, with equal or signature-similar declarations, is a
    relocation, not a removal+addition. Emits one ``relocated`` change carrying
    the OLD id (so the ledger marks the origin entry relocated) and the new id
    in ``relocated_to``. Mutates ``removed`` / ``added`` in place, consuming any
    paired ids. The same-``rest`` requirement (not just the same leaf) prevents
    the ``Foo::new`` / ``Bar::new`` constructor collision from pairing.
    """
    changes: list[SurfaceChange] = []
    for removed_id in sorted(removed):
        rsym = removed[removed_id]
        if rsym.surface != "rust":
            continue
        r_crate, r_rest = _rust_suffix(removed_id)
        if not r_rest:
            continue
        removed_sig = canonicalize_signature(rsym.signature)
        best_id, best_score = "", -1.0
        for cand_id, cand in added.items():
            if cand.surface != "rust":
                continue
            c_crate, c_rest = _rust_suffix(cand_id)
            if c_rest != r_rest or c_crate == r_crate:
                continue
            cand_sig = canonicalize_signature(cand.signature)
            score = (
                1.0
                if cand_sig == removed_sig
                else SequenceMatcher(None, removed_sig, cand_sig).ratio()
            )
            if score > best_score:
                best_id, best_score = cand_id, score
        if not best_id or best_score < RENAME_SIMILARITY_THRESHOLD:
            continue
        new_sym = added[best_id]
        changes.append(
            SurfaceChange(
                change_type="relocated",
                id=removed_id,
                surface="rust",
                component=new_sym.component or rsym.component,
                from_signature=removed_sig,
                to_signature=canonicalize_signature(new_sym.signature),
                summary=f"Relocated {removed_id} -> {best_id}",
                confidence="medium",
                source="diff",
                similarity=round(best_score, 3),
                relocated_to=best_id,
            )
        )
        del removed[removed_id]
        del added[best_id]
    return changes


def _group_by_named_container(
    symbols: dict[str, SurfaceSymbol],
) -> dict[str, dict[str, str]]:
    """Map each *named* container to its ``{member_leaf: id}`` for its members.

    Only containers that themselves have a parent are kept, so a class or
    struct (e.g. ``python:pkg.mod.Class``) groups its members while a top-level
    module (``python:pkg.mod``) does not — a whole-module rename is out of
    scope and far more likely to be a coverage artifact than a real rename.

    Rust ids are excluded: rust has a stricter same-path relocation detector
    (:func:`_detect_relocations`), and rust structs that implement a shared
    trait expose the same method leaves (``new`` / ``free`` / ``add_request``),
    so a member-leaf-overlap heuristic fabricates moves between distinct types
    (e.g. ``KvRouter`` <-> ``WorkerSelector``). Container-rename detection is
    therefore reserved for the structured non-rust surfaces (python / crd /
    config), which have no such trait-method aliasing.
    """
    groups: dict[str, dict[str, str]] = {}
    for sid in symbols:
        if sid.startswith("rust:"):
            continue
        container, leaf = _split_container(sid)
        if container is None or _rename_container(container) is None:
            continue
        groups.setdefault(container, {})[leaf] = sid
    return groups


def _detect_container_renames(
    removed: dict[str, SurfaceSymbol], added: dict[str, SurfaceSymbol]
) -> list[SurfaceChange]:
    """Pair members of a renamed container (class / struct) as relocations.

    A leaf rename (``_detect_renames``) requires the container to match, so it
    cannot recognize that ``backend.EngineConfig.*`` became
    ``backend.LlmRegistration.*``: every member's container changed in lockstep.
    This pass finds a removed container and an added sibling container (same
    parent, different leaf, same surface) whose member leaves overlap past the
    dual :data:`CONTAINER_RENAME_MIN_SHARED` / :data:`CONTAINER_RENAME_OVERLAP`
    guard, then emits one ``relocated`` change per shared member (old id ->
    new id) so the ledger marks them moved, not removed+added. Members present
    on only one side stay in ``removed`` / ``added`` as genuine drops / adds.
    Mutates ``removed`` / ``added`` in place, consuming paired ids.
    """
    rem_groups = _group_by_named_container(removed)
    add_groups = _group_by_named_container(added)
    changes: list[SurfaceChange] = []
    consumed: set[str] = set()
    for old_container in sorted(rem_groups):
        old_members = rem_groups[old_container]
        old_parent, old_leaf = _split_container(old_container)
        old_surface = removed[next(iter(old_members.values()))].surface
        best_container: str = ""
        best_shared: list[str] = []
        for new_container, new_members in add_groups.items():
            if new_container in consumed:
                continue
            new_parent, new_leaf = _split_container(new_container)
            if new_parent != old_parent or new_leaf == old_leaf:
                continue
            if added[next(iter(new_members.values()))].surface != old_surface:
                continue
            shared = [leaf for leaf in old_members if leaf in new_members]
            if len(shared) > len(best_shared):
                best_container, best_shared = new_container, shared
        if not best_container:
            continue
        new_members = add_groups[best_container]
        overlap = len(best_shared) / min(len(old_members), len(new_members))
        if (
            len(best_shared) < CONTAINER_RENAME_MIN_SHARED
            or overlap < CONTAINER_RENAME_OVERLAP
        ):
            continue
        consumed.add(best_container)
        for leaf in sorted(best_shared):
            old_id, new_id = old_members[leaf], new_members[leaf]
            osym, nsym = removed[old_id], added[new_id]
            changes.append(
                SurfaceChange(
                    change_type="relocated",
                    id=old_id,
                    surface=osym.surface,
                    component=nsym.component or osym.component,
                    from_signature=canonicalize_signature(osym.signature),
                    to_signature=canonicalize_signature(nsym.signature),
                    summary=f"Relocated {old_id} -> {new_id}",
                    confidence="medium",
                    source="diff",
                    similarity=round(overlap, 3),
                    relocated_to=new_id,
                )
            )
            del removed[old_id]
            del added[new_id]
    return changes


def _detect_crate_merges(
    old_symbols: list[SurfaceSymbol],
    new_symbols: list[SurfaceSymbol],
    added: dict[str, SurfaceSymbol],
) -> list[SurfaceChange]:
    """Pair symbols of a merged-away rust crate with their new home.

    For each ``CRATE_MERGES`` entry, a symbol ``rust:<old>::<rest>`` present in
    ``old_symbols`` whose merged id ``rust:<new>::<prefix>::<rest>`` is present
    in ``new_symbols`` is emitted as one ``relocated`` change. The new id is
    discarded from ``added`` so a relocated-in symbol is not also reported as a
    bare addition. Operates on the *ungated* symbol lists because the per-crate
    coverage gate hides the old crate. Conservative: an old symbol with no
    merged match is left untouched (still gated out, never a false removal).
    """
    if not CRATE_MERGES:
        return []
    old_by_id = {s.id: s for s in old_symbols}
    new_by_id = {s.id: s for s in new_symbols}
    changes: list[SurfaceChange] = []
    for old_crate, (new_crate, prefix) in CRATE_MERGES.items():
        old_prefix = f"rust:{old_crate}::"
        for old_id in sorted(old_by_id):
            if not old_id.startswith(old_prefix):
                continue
            rest = old_id[len(old_prefix) :]
            new_id = f"rust:{new_crate}::{prefix}::{rest}"
            nsym = new_by_id.get(new_id)
            if nsym is None:
                continue
            osym = old_by_id[old_id]
            changes.append(
                SurfaceChange(
                    change_type="relocated",
                    id=old_id,
                    surface="rust",
                    component=nsym.component or osym.component,
                    from_signature=canonicalize_signature(osym.signature),
                    to_signature=canonicalize_signature(nsym.signature),
                    summary=f"Relocated {old_id} -> {new_id}",
                    confidence="medium",
                    source="diff",
                    relocated_to=new_id,
                )
            )
            added.pop(new_id, None)
    return changes


def _detect_renames(
    removed: dict[str, SurfaceSymbol], added: dict[str, SurfaceSymbol]
) -> list[SurfaceChange]:
    """Pair removed+added symbols with similar signatures as single renames.

    Mutates ``removed`` and ``added`` in place, deleting any paired ids so the
    caller emits only the genuinely added/removed remainder.
    """
    changes: list[SurfaceChange] = []
    for removed_id in sorted(removed):
        match_id, score = _best_rename_match(removed[removed_id], added)
        if not match_id or score < RENAME_SIMILARITY_THRESHOLD:
            continue
        new_sym = added[match_id]
        changes.append(
            SurfaceChange(
                change_type="renamed_function",
                id=removed_id,
                surface=new_sym.surface,
                component=new_sym.component,
                from_signature=canonicalize_signature(removed[removed_id].signature),
                to_signature=canonicalize_signature(new_sym.signature),
                summary=f"Renamed {removed_id} -> {new_sym.id}",
                confidence="medium",
                source="diff",
                similarity=round(score, 3),
                relocated_to=new_sym.id,
            )
        )
        del removed[removed_id]
        del added[match_id]
    return changes


def diff_snapshots(old: SurfaceSnapshot, new: SurfaceSnapshot) -> OperationResult:
    """Diff two snapshots into a sorted list of SurfaceChange events.

    Returns an :class:`OperationResult` whose ``data`` carries:

    - ``changes``: ``list[dict]`` of :meth:`SurfaceChange.to_dict`, sorted
    - ``coverage_gaps``: surfaces not comparable (absent in either snapshot)

    Only surfaces covered in *both* snapshots are compared.
    """
    start = time.time()
    surfaces = _covered_surfaces(old, new)
    rust_crates = _covered_rust_crates(old, new)
    old_idx = _index(old.symbols, surfaces, rust_crates)
    new_idx = _index(new.symbols, surfaces, rust_crates)

    changes: list[SurfaceChange] = []
    for sid in old_idx.keys() & new_idx.keys():
        change = _signature_change(old_idx[sid], new_idx[sid])
        if change is not None:
            changes.append(change)

    removed = {sid: old_idx[sid] for sid in old_idx.keys() - new_idx.keys()}
    added = {sid: new_idx[sid] for sid in new_idx.keys() - old_idx.keys()}
    # Crate merges first (they read the ungated full sets and prune `added`),
    # then rust crate-move relocations, then container renames (a class /
    # struct renamed in lockstep), then leaf renames on the remainder, so a
    # moved symbol is not also offered to a later, weaker matcher.
    changes.extend(_detect_crate_merges(old.symbols, new.symbols, added))
    changes.extend(_detect_relocations(removed, added))
    changes.extend(_detect_container_renames(removed, added))
    changes.extend(_detect_renames(removed, added))

    for sid in sorted(removed):
        sym = removed[sid]
        changes.append(
            SurfaceChange(
                change_type="removed",
                id=sid,
                surface=sym.surface,
                component=sym.component,
                from_signature=canonicalize_signature(sym.signature),
                summary=f"Removed {sid}",
                confidence="high",
                source="diff",
            )
        )
    for sid in sorted(added):
        sym = added[sid]
        changes.append(
            SurfaceChange(
                change_type="added",
                id=sid,
                surface=sym.surface,
                component=sym.component,
                to_signature=canonicalize_signature(sym.signature),
                summary=f"Added {sid}",
                confidence="high",
                source="diff",
            )
        )

    changes.sort(key=lambda c: (c.surface, c.id, c.change_type))
    all_surfaces = set(old.coverage) | set(new.coverage)
    coverage_gaps = sorted(all_surfaces - surfaces)

    result = OperationResult(
        data={
            "changes": [c.to_dict() for c in changes],
            "coverage_gaps": coverage_gaps,
        },
        metadata={
            "from_release": old.release,
            "to_release": new.release,
            "change_count": len(changes),
            "compared_surfaces": sorted(surfaces),
        },
    )
    result.add_timing(start)
    return result
