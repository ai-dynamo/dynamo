# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rust public-surface extractor (no-build, regex + brace scan).

Two slices, both emitting the ``rust`` surface:

- **C ABI** (high confidence): the sanctioned, stable C boundary in
  ``lib/bindings/c``. ``#[no_mangle] pub extern "C" fn`` exports and
  ``#[repr(C)] pub`` types are a deliberate, documented public contract, so
  these land ``confidence=high``, ``provenance=c_abi``.
- **Curated internal libs** (medium confidence): a small allowlist of
  Dynamo-authored ``lib/<dir>`` crates (the vendored ``async-openai`` fork is
  excluded). Items reachable through ``pub mod`` chains from the crate root
  ``lib.rs`` are emitted with their FULL (multi-line) signatures. ``pub``-ness
  is a coarse stability proxy, so these land ``confidence=medium``.

Each symbol carries ``metadata['declared_public']`` -- the positive signal the
stability layer (:mod:`..stability`) uses to gate the ``stable`` tier. The C ABI
and unpacked crates.io releases are declared public by construction (a real
stability contract); an internal curated ``lib/<dir>`` crate declares only its
crate-root facade (root-defined items + crate-root ``pub use`` re-exports, both
emitted with an empty module path). A merely-reachable deep item is therefore
floored at ``experimental`` so internal churn is informational, not a broken
promise. See :meth:`_Emitter._declared_public`.

No Rust toolchain is invoked: this is a text scan, bounded the same way as the
marker scanner (build/vendor dirs pruned, oversize files skipped, read errors
appended as :class:`OpError` rather than raised). It is therefore best-effort,
not a rustdoc-accurate surface; the accuracy gate in the audit bounds it.

Symbol ids follow the frozen grammar (one form per surface)::

    rust:<crate>::<module::path>::<item>          # free fn / type / const ...
    rust:<crate>::<module::path>::<Type>::<method>  # inherent-impl method

The crate name and file-level module path are derived by :func:`rust_crate_for`
and :func:`rust_module_path_for`, which :mod:`markers` imports too so a
``#[deprecated]`` marker and this extractor address the same symbol by the same
id (the whole point of the shared helpers).

Per-crate coverage is reported in ``metadata['coverage_detail']`` keyed
``rust:<crate>``: a curated crate absent at a given ref is recorded NOT-covered
so the diff engine skips it rather than reading a crate that simply moved or
was renamed as a wave of removals.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from api_surface.models import SurfaceSymbol
from api_surface.results import OperationResult, OpError

SURFACE = "rust"

_OPERATION = "api_surface.extract.rust"
_MAX_FILE_BYTES = 1_000_000

# Curated, Dynamo-authored, library-consumed crates (directory names under
# ``lib/``). The vendored ``async-openai`` fork is intentionally excluded.
CURATED_LIB_DIRS: tuple[str, ...] = (
    "llm",
    "runtime",
    "protocols",  # extracted to ai-dynamo/frontend-crates post-v1.2.0
    "parsers",  # extracted to ai-dynamo/frontend-crates post-v1.2.0
    "tokenizers",  # extracted to ai-dynamo/frontend-crates post-v1.2.0
    "backend-common",  # the unified backend abstraction, consumed by the python bindings
    "tokens",  # tokenization, consumed by llm + kv-router
    "rl",  # consumed by llm
    "kv-router",
    "memory",
    "kvbm-common",
    "kvbm-config",
    "kvbm-consolidator",
    "kvbm-engine",
    "kvbm-kernels",
    "kvbm-logical",
    "kvbm-physical",
)

# Never scanned even if present under ``lib/`` (vendored / third-party forks).
_EXCLUDED_LIB_DIRS = frozenset({"async-openai"})

# C ABI binding crate location (relative to repo root) + its scanned src.
_C_ABI_DIR = "lib/bindings/c"


# =============================================================================
# Shared id grammar helpers (imported by markers.py for reconciliation)
# =============================================================================


def rust_id(crate: str, module_path: str, item: str) -> str:
    """Build a ``rust:<crate>::<module::path>::<item>`` id.

    ``module_path`` may be empty (crate-root item), in which case the id
    collapses to ``rust:<crate>::<item>``. All three parts are whitespace-
    stripped and any leading/trailing ``::`` on ``module_path`` is removed so
    the grammar stays canonical regardless of how callers assemble the path.
    """
    crate = crate.strip()
    item = item.strip()
    module_path = module_path.strip().strip(":")
    if module_path:
        return f"rust:{crate}::{module_path}::{item}"
    return f"rust:{crate}::{item}"


def rust_crate_for(file_path: Path, repo_path: Path) -> str:
    """Best-effort crate name for a Rust file path.

    Shared with :mod:`markers` so a marker and an extracted symbol resolve to
    the same crate. Heuristic order: under ``lib/<dir>/`` -> ``dynamo-<dir>``;
    otherwise the directory immediately above ``src/``; otherwise the file's
    parent directory name; finally ``"unknown"``.
    """
    try:
        rel = file_path.resolve().relative_to(repo_path.resolve())
    except ValueError:
        return file_path.parent.name or "unknown"
    parts = rel.parts
    if "lib" in parts:
        idx = parts.index("lib")
        if idx + 1 < len(parts):
            return f"dynamo-{parts[idx + 1]}"
    if "src" in parts:
        idx = parts.index("src")
        if idx > 0:
            return parts[idx - 1]
    return file_path.parent.name or "unknown"


def rust_module_path_for(file_path: Path, repo_path: Path) -> str:
    """File-level Rust module path (``component::endpoint``), or ``""`` at root.

    Derived from the path segments after ``src``: the file stem is appended
    unless it is a module root (``lib``/``mod``/``main``), which collapses to
    its directory. This is the file-scope module only; inline ``pub mod``
    nesting is layered on by the extractor's body scan, and markers rely on the
    ledger's ``(crate, trailing-name)`` reconcile for inline-nested items.
    """
    try:
        rel = file_path.resolve().relative_to(repo_path.resolve())
    except ValueError:
        rel = file_path
    parts = list(rel.parts)
    if "src" not in parts:
        return ""
    tail = parts[parts.index("src") + 1 :]
    if not tail:
        return ""
    stem = Path(tail[-1]).stem
    segments = tail[:-1] if stem in {"lib", "mod", "main"} else [*tail[:-1], stem]
    return "::".join(segments)


# =============================================================================
# Low-level text scanning (strings / comments / braces)
# =============================================================================


def _skip_ws_comments(text: str, i: int, end: int) -> int:
    """Advance past whitespace, ``//`` line comments and ``/* */`` blocks."""
    while i < end:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if text.startswith("//", i):
            nl = text.find("\n", i)
            i = end if nl == -1 else nl + 1
            continue
        if text.startswith("/*", i):
            close = text.find("*/", i + 2)
            i = end if close == -1 else close + 2
            continue
        break
    return i


def _skip_char_or_lifetime(text: str, i: int) -> int:
    """Advance past a char literal (``'a'`` / ``'\\n'``) or a lifetime (``'static``).

    A Rust ``'`` is ambiguous: it opens a char literal OR a lifetime / loop
    label. A char literal is ``'`` then an escape (``'\\x'``) or exactly one
    char (``'a'``) closed by ``'``; anything else (``'static``, ``'a,``) is a
    lifetime, where the ``'`` is not a quote and only that one char is consumed.
    Treating a lifetime as a char-literal opener would skip to the next ``'`` and
    swallow whole item bodies (and break brace matching), so this distinction is
    load-bearing for signature capture.
    """
    n = len(text)
    if i + 1 < n and text[i + 1] == "\\":
        j = i + 2
        while j < n and text[j] != "'":
            j += 1
        return j + 1 if j < n else n
    if i + 2 < n and text[i + 2] == "'":
        return i + 3
    return i + 1


def _skip_string(text: str, i: int) -> int:
    """Advance past a Rust string / char / raw-string literal / lifetime at ``i``."""
    n = len(text)
    # Raw string: r"..." / r#"..."# / r##"..."## ...
    if text[i] == "r" and i + 1 < n and text[i + 1] in '#"':
        j = i + 1
        hashes = 0
        while j < n and text[j] == "#":
            hashes += 1
            j += 1
        if j < n and text[j] == '"':
            close = '"' + "#" * hashes
            end = text.find(close, j + 1)
            return n if end == -1 else end + len(close)
    if text[i] == "'":
        return _skip_char_or_lifetime(text, i)
    quote = text[i]
    j = i + 1
    while j < n:
        if text[j] == "\\":
            j += 2
            continue
        if text[j] == quote:
            return j + 1
        j += 1
    return n


def _matching_brace(text: str, open_idx: int) -> int:
    """Index of the ``}`` matching the ``{`` at ``open_idx`` (string/comment aware)."""
    depth = 0
    i = open_idx
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in "\"'":
            i = _skip_string(text, i)
            continue
        if text.startswith("//", i):
            nl = text.find("\n", i)
            i = n if nl == -1 else nl + 1
            continue
        if text.startswith("/*", i):
            close = text.find("*/", i + 2)
            i = n if close == -1 else close + 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return n - 1


def _header_end(text: str, start: int, end: int) -> tuple[int, str]:
    """Find the item header terminator from ``start``.

    Returns ``(index, terminator)`` where terminator is ``"{"`` (item has a
    braced body) or ``";"`` (statement item), scanning at paren/bracket/angle
    depth zero so a ``{`` inside generics or a ``where`` bound is not mistaken
    for the body. Returns ``(end, "")`` if neither is found.
    """
    i = start
    paren = bracket = angle = 0
    while i < end:
        ch = text[i]
        if ch in "\"'":
            i = _skip_string(text, i)
            continue
        if text.startswith("//", i):
            nl = text.find("\n", i)
            i = end if nl == -1 else nl + 1
            continue
        if text.startswith("/*", i):
            close = text.find("*/", i + 2)
            i = end if close == -1 else close + 2
            continue
        if ch == "(":
            paren += 1
        elif ch == ")":
            paren = max(0, paren - 1)
        elif ch == "[":
            bracket += 1
        elif ch == "]":
            bracket = max(0, bracket - 1)
        elif ch == "<":
            angle += 1
        elif ch == ">":
            angle = max(0, angle - 1)
        elif paren == 0 and bracket == 0:
            if ch == "{":
                return i, "{"
            if ch == ";" and angle == 0:
                return i, ";"
        i += 1
    return end, ""


# =============================================================================
# Item recognition
# =============================================================================

_IDENT = r"[A-Za-z_][A-Za-z0-9_]*"

# An item head at the current scan position. Captures visibility (to test for
# `pub`) and the introducing keyword. `pub(crate)` / `pub(super)` / `pub(in ..)`
# are captured in `restricted` and treated as NOT public.
_ITEM_HEAD = re.compile(
    r"""
    (?P<vis>pub\s*(?:\(\s*(?P<restricted>[^)]*)\s*\)\s*)?)?
    (?:default\s+)?
    (?:async\s+)?
    (?:unsafe\s+)?
    (?:extern\s+"[^"]*"\s+)?
    (?:const\s+)?
    (?P<kw>fn|struct|enum|trait|type|static|const|union|mod|impl)
    (?![A-Za-z0-9_])
    """,
    re.VERBOSE,
)

_NAME_AFTER_KW = re.compile(rf"\s+(?P<name>{_IDENT})")

# Kinds that introduce a braced body we must skip (we emit the header only).
_BRACED_TYPES = frozenset({"struct", "enum", "trait", "union"})


def _is_public(match: re.Match[str]) -> bool:
    """True iff the matched item head is unconditionally ``pub``.

    ``pub`` with no restriction is public; ``pub(crate)`` / ``pub(super)`` /
    ``pub(in ...)`` are not part of the external surface; no ``pub`` at all is
    private.
    """
    if match.group("vis") is None:
        return False
    return match.group("restricted") is None


def _canon(s: str) -> str:
    """Collapse whitespace runs to single spaces and trim."""
    return " ".join(s.split())


def _strip_leading_generics(body: str) -> str:
    """Drop a leading ``<...>`` generic parameter list (the ``impl<T>`` part).

    Returns ``body`` unchanged when it does not open with ``<``. Brace-matched
    so nested generics (``impl<T: Into<U>>``) are consumed whole.
    """
    body = body.lstrip()
    if not body.startswith("<"):
        return body
    depth = 0
    for j, ch in enumerate(body):
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
            if depth == 0:
                return body[j + 1 :]
    return body


def _impl_target(header: str) -> str | None:
    """Return the inherent-impl target type name, or ``None`` for a trait impl.

    ``impl<T> Foo<T>`` -> ``Foo``; ``impl path::to::Bar`` -> ``Bar``;
    ``impl Trait for Baz`` -> ``None`` (trait impls re-state the trait's
    surface, not new inherent API). Generic parameter lists and a trailing
    ``where`` clause (single- or multi-line) are stripped before the type path
    is read, so ``impl<P, C> Queue<P, C>\\nwhere P: Bound`` resolves to
    ``Queue`` rather than capturing the where-clause as the type name.
    """
    # Drop the ``impl`` keyword and any leading ``impl<...>`` generic list so
    # the remainder begins at the type path (inherent) or trait path (trait).
    body = _strip_leading_generics(header[len("impl") :]).strip()
    # A top-level ` for ` marks a trait impl; a top-level ` where ` ends the
    # type path. Track ``<>`` depth so generic args do not trip either guard.
    depth = 0
    for m in re.finditer(r"<|>|\bfor\b|\bwhere\b", body):
        tok = m.group(0)
        if tok == "<":
            depth += 1
        elif tok == ">":
            depth = max(0, depth - 1)
        elif depth == 0 and tok == "for":
            return None
        elif depth == 0 and tok == "where":
            body = body[: m.start()]
            break
    # Strip the type's own generics, then keep the final path segment.
    target = re.sub(r"<.*", "", body, flags=re.DOTALL).strip()
    target = target.split("::")[-1].strip()
    # A real type is a bare identifier; anything else is a parse artifact and is
    # dropped rather than emitted as a corrupt id.
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", target):
        return None
    return target


class _Emitter:
    """Accumulates :class:`SurfaceSymbol` dicts for one crate."""

    def __init__(self, crate: str, provenance: str, confidence: str) -> None:
        """Initialize one crate-scoped symbol emitter."""
        self.crate = crate
        self.provenance = provenance
        self.confidence = confidence
        self.symbols: list[dict[str, Any]] = []
        self._seen: set[str] = set()

    def _declared_public(self, module_path: str) -> bool:
        """Positive public-API signal for a Rust symbol (the declared-public gate).

        A sanctioned/published boundary is declared public by construction: the
        C ABI (``provenance=c_abi``) and unpacked crates.io releases
        (``provenance=crates_io``) carry a real stability contract, so every
        symbol they expose is declared. An *internal* curated workspace crate
        (``provenance=lib``) has no semver promise -- ``pub``-ness there is just
        a reachability proxy -- so only its crate-root *facade* counts: a symbol
        defined at the root or surfaced by a crate-root ``pub use`` re-export
        (both emitted with an empty ``module_path``). A merely-reachable deep
        item (``approx::PruneManager``, ``bench_utils::*``) is *not* declared,
        so the stability layer floors it at ``experimental`` -- internal churn,
        not a broken promise.
        """
        return self.provenance != "lib" or module_path == ""

    def add(self, module_path: str, item: str, kind: str, signature: str) -> None:
        """Add one canonical, de-duplicated Rust symbol."""
        sid = rust_id(self.crate, module_path, item)
        if sid in self._seen:
            return
        self._seen.add(sid)
        self.symbols.append(
            SurfaceSymbol(
                surface=SURFACE,
                kind=kind,
                id=sid,
                signature=_canon(signature),
                metadata={
                    "provenance": self.provenance,
                    "confidence": self.confidence,
                    "declared_public": self._declared_public(module_path),
                },
            ).to_dict()
        )


# A module-level ``pub use`` re-export head (NOT ``pub(crate) use`` -- the
# intervening ``(`` defeats ``pub\s+use``, so restricted re-exports are excluded,
# matching the visibility rule applied to items).
_PUB_USE_HEAD = re.compile(r"pub\s+use\s+")


def _split_top_level_commas(s: str) -> list[str]:
    """Split a use-tree group body on commas that are not inside ``{...}``."""
    parts: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in s:
        if ch == "{":
            depth += 1
            cur.append(ch)
        elif ch == "}":
            depth -= 1
            cur.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    tail = "".join(cur)
    if tail.strip():
        parts.append(tail)
    return parts


def _use_tree_leaves(tree: str, in_group: bool = False) -> list[str]:
    """Names a ``pub use <tree>`` exposes at the re-exporting module.

    ``foo::Bar`` -> ``[Bar]``; ``foo::Bar as Baz`` -> ``[Baz]``;
    ``foo::{A, b::C, D as E}`` -> ``[A, C, E]``; nested groups recurse. Globs
    (``foo::*``) yield nothing (unenumerable without resolution); a bare
    top-level module re-export (``pub use foo;``) yields nothing (no item).
    ``in_group`` marks recursion inside ``{...}``, where a bare ident (the brace
    prefix already consumed) is itself an exposed item.
    """
    tree = tree.strip()
    if not tree:
        return []
    brace = tree.find("{")
    if brace != -1:
        depth = 0
        close = -1
        for j in range(brace, len(tree)):
            if tree[j] == "{":
                depth += 1
            elif tree[j] == "}":
                depth -= 1
                if depth == 0:
                    close = j
                    break
        if close != -1:
            leaves: list[str] = []
            for part in _split_top_level_commas(tree[brace + 1 : close]):
                leaves.extend(_use_tree_leaves(part, in_group=True))
            return leaves
    alias = re.search(r"\bas\s+([A-Za-z_][A-Za-z0-9_]*)\s*$", tree)
    if alias:
        return [alias.group(1)]
    last = tree.split("::")[-1].strip()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", last):
        return []
    if "::" in tree or in_group:
        return [last]
    return []


def _scan_region(
    text: str, start: int, end: int, module_path: str, emitter: _Emitter
) -> None:
    """Scan ``text[start:end]`` (one module body) for ``pub`` items.

    Recurses into inline ``pub mod`` blocks (appending to ``module_path``) and
    inherent ``impl`` blocks (emitting ``Type::method`` ids). Public ``pub use``
    re-exports are emitted at this module's path (the facade idiom: a private
    ``mod`` re-exported via ``pub use`` -- its types' only public path is here).
    Private items, private/inline ``mod`` blocks, function bodies and trait
    impls are skipped.
    """
    i = start
    while i < end:
        i = _skip_ws_comments(text, i, end)
        if i >= end:
            break
        ch = text[i]
        if ch in "\"'":
            i = _skip_string(text, i)
            continue
        if ch == "{":
            i = _matching_brace(text, i) + 1
            continue
        if ch == "}":
            break
        if ch == "#":  # attribute: skip `#[...]` (and `#![...]`)
            br = text.find("[", i)
            if br != -1 and br - i <= 2:
                i = _matching_bracket(text, br) + 1
                continue
        use_m = _PUB_USE_HEAD.match(text, i)
        if use_m:
            semi = text.find(";", use_m.end())
            if semi == -1 or semi >= end:
                semi = end
            tree = text[use_m.end() : semi]
            signature = f"pub use {_canon(tree)}"
            for leaf in _use_tree_leaves(tree):
                emitter.add(module_path, leaf, "reexport", signature)
            i = semi + 1
            continue
        match = _ITEM_HEAD.match(text, i)
        if not match:
            i = _advance_token(text, i, end)
            continue
        i = _handle_item(text, match, end, module_path, emitter)


def _matching_bracket(text: str, open_idx: int) -> int:
    """Index of the ``]`` matching the ``[`` at ``open_idx`` (string aware)."""
    depth = 0
    i = open_idx
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in "\"'":
            i = _skip_string(text, i)
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return n - 1


def _advance_token(text: str, i: int, end: int) -> int:
    """Advance past one identifier run or a single non-word char (progress)."""
    m = re.match(rf"{_IDENT}", text[i:end])
    if m:
        return i + m.end()
    return i + 1


def _handle_item(
    text: str, match: re.Match[str], end: int, module_path: str, emitter: _Emitter
) -> int:
    """Process one matched item head; return the index to continue scanning."""
    kw = match.group("kw")
    is_pub = _is_public(match)
    head_start, hdr_term_search = match.start(), match.end()
    term_idx, term = _header_end(text, hdr_term_search, end)
    header = text[head_start:term_idx]

    if kw == "mod":
        return _handle_mod(text, header, is_pub, term_idx, term, module_path, emitter)
    if kw == "impl":
        return _handle_impl(text, header, term_idx, term, module_path, emitter)

    name_m = _NAME_AFTER_KW.match(text, match.end())
    if not name_m:
        return _skip_body_or_stmt(text, term_idx, term)
    if is_pub:
        emitter.add(module_path, name_m.group("name"), _kind_for(kw), header)
    return _skip_body_or_stmt(text, term_idx, term)


def _kind_for(kw: str) -> str:
    """Map a Rust keyword to a SurfaceSymbol kind."""
    return {
        "fn": "function",
        "struct": "struct",
        "enum": "enum",
        "trait": "trait",
        "union": "union",
        "type": "type_alias",
        "const": "const",
        "static": "static",
    }.get(kw, kw)


def _skip_body_or_stmt(text: str, term_idx: int, term: str) -> int:
    """Advance past a braced body or to just after the ``;`` terminator."""
    if term == "{":
        return _matching_brace(text, term_idx) + 1
    if term == ";":
        return term_idx + 1
    return term_idx + 1


def _handle_mod(
    text: str,
    header: str,
    is_pub: bool,
    term_idx: int,
    term: str,
    module_path: str,
    emitter: _Emitter,
) -> int:
    """Recurse an inline ``pub mod NAME { ... }``; skip private/file mods here."""
    if term != "{":
        # `mod NAME;` file declaration: handled by the file-reachability walk.
        return term_idx + 1
    close = _matching_brace(text, term_idx)
    if is_pub:
        name_m = re.search(rf"\bmod\s+({_IDENT})", header)
        if name_m:
            child = (
                f"{module_path}::{name_m.group(1)}" if module_path else name_m.group(1)
            )
            _scan_region(text, term_idx + 1, close, child, emitter)
    return close + 1


def _handle_impl(
    text: str,
    header: str,
    term_idx: int,
    term: str,
    module_path: str,
    emitter: _Emitter,
) -> int:
    """Emit ``pub`` methods of an inherent ``impl Type { ... }`` block."""
    if term != "{":
        return term_idx + 1
    close = _matching_brace(text, term_idx)
    target = _impl_target(header)
    if target is None:
        return close + 1
    method_module = f"{module_path}::{target}" if module_path else target
    _scan_impl_methods(text, term_idx + 1, close, method_module, emitter)
    return close + 1


def _scan_impl_methods(
    text: str, start: int, end: int, module_path: str, emitter: _Emitter
) -> None:
    """Emit ``pub fn`` methods inside an impl body (skips bodies + non-pub)."""
    i = start
    while i < end:
        i = _skip_ws_comments(text, i, end)
        if i >= end:
            break
        ch = text[i]
        if ch in "\"'":
            i = _skip_string(text, i)
            continue
        if ch == "{":
            i = _matching_brace(text, i) + 1
            continue
        if ch == "}":
            break
        if ch == "#":
            br = text.find("[", i)
            if br != -1 and br - i <= 2:
                i = _matching_bracket(text, br) + 1
                continue
        match = _ITEM_HEAD.match(text, i)
        if not match or match.group("kw") != "fn":
            i = _advance_token(text, i, end)
            continue
        term_idx, term = _header_end(text, match.end(), end)
        header = text[match.start() : term_idx]
        name_m = _NAME_AFTER_KW.match(text, match.end())
        if name_m and _is_public(match):
            emitter.add(module_path, name_m.group("name"), "method", header)
        i = _skip_body_or_stmt(text, term_idx, term)


# =============================================================================
# File-module reachability (pub mod chains from the crate root)
# =============================================================================

_PUB_MOD_DECL = re.compile(rf"^\s*pub\s+mod\s+({_IDENT})\s*;", re.MULTILINE)


def _resolve_child_module(current: Path, name: str) -> Path | None:
    """Resolve a ``pub mod NAME;`` to its file, or ``None`` if absent.

    A root file (``lib.rs`` / ``mod.rs``) hosts submodules in its own dir; a
    plain ``foo.rs`` hosts them in a sibling ``foo/`` dir (Rust 2018 layout).
    """
    base = (
        current.parent
        if current.stem in {"lib", "mod"}
        else current.parent / current.stem
    )
    for candidate in (base / f"{name}.rs", base / name / "mod.rs"):
        if candidate.is_file():
            return candidate
    return None


def _reachable_modules(crate_src: Path) -> list[tuple[Path, str]]:
    """Return ``(file, module_path)`` reachable via ``pub mod`` from ``lib.rs``."""
    root = crate_src / "lib.rs"
    if not root.is_file():
        return []
    out: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    stack: list[tuple[Path, str]] = [(root, "")]
    while stack:
        file_path, module_path = stack.pop()
        resolved = file_path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append((file_path, module_path))
        text = _read(file_path)
        if text is None:
            continue
        for m in _PUB_MOD_DECL.finditer(text):
            child = _resolve_child_module(file_path, m.group(1))
            if child is None:
                continue
            child_path = f"{module_path}::{m.group(1)}" if module_path else m.group(1)
            stack.append((child, child_path))
    return out


def _read(path: Path) -> str | None:
    """Read a bounded text file, or ``None`` on oversize / error."""
    try:
        if path.stat().st_size > _MAX_FILE_BYTES:
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


# =============================================================================
# C ABI slice
# =============================================================================

# The ``no_mangle`` attribute appears both bare (``#[no_mangle]``) and wrapped
# in the Rust 2024 unsafe-attribute form (``#[unsafe(no_mangle)]``); match any
# attribute whose body mentions ``no_mangle``. The exported fn may carry
# ``pub``/``unsafe`` qualifiers before ``extern "C"``.
_C_EXPORT_FN = re.compile(
    rf'#\[[^\]]*\bno_mangle\b[^\]]*\][^;{{}}]*?extern\s+"C"\s+fn\s+({_IDENT})',
    re.DOTALL,
)
_C_REPR_TYPE = re.compile(
    rf"#\[[^\]]*\brepr\s*\(\s*C\b[^\]]*\][^;{{}}]*?pub\s+(?:struct|enum|union)\s+({_IDENT})",
    re.DOTALL,
)


def _extract_c_abi(
    repo_path: Path, crate: str, emitter: _Emitter, result: OperationResult
) -> bool:
    """Scan ``lib/bindings/c`` for the exported C ABI; return whether covered."""
    c_dir = repo_path / _C_ABI_DIR
    if not c_dir.is_dir():
        return False
    covered = False
    for rs in sorted((c_dir / "src").rglob("*.rs")) if (c_dir / "src").is_dir() else []:
        text = _read(rs)
        if text is None:
            result.errors.append(
                OpError(
                    operation=_OPERATION,
                    message=f"C ABI source unreadable: {rs}",
                    details={"path": str(rs)},
                )
            )
            continue
        covered = True
        for m in _C_EXPORT_FN.finditer(text):
            emitter.add("", m.group(1), "c_function", f'extern "C" fn {m.group(1)}')
        for m in _C_REPR_TYPE.finditer(text):
            emitter.add("", m.group(1), "c_type", f"repr(C) {m.group(1)}")
    return covered


# =============================================================================
# Public entry point
# =============================================================================


def scan_crate_src(
    crate_src: Path,
    crate: str,
    provenance: str,
    confidence: str,
    result: OperationResult,
) -> list[dict[str, Any]] | None:
    """Scan one crate ``src`` dir as ``crate``, returning its public symbols.

    Walks the ``pub mod`` reachability chain from ``lib.rs`` exactly like the
    curated-lib pass, emitting ids ``rust:<crate>::...``. Returns ``None`` when
    ``crate_src`` is not a directory (the caller records coverage ``False``);
    unreadable files are logged on ``result`` and skipped. ``crate`` is supplied
    explicitly (not path-derived) so a crate scanned from an arbitrary location
    -- e.g. an unpacked crates.io tarball -- still produces monorepo-identical
    ids.
    """
    if not crate_src.is_dir():
        return None
    emitter = _Emitter(crate, provenance=provenance, confidence=confidence)
    for file_path, module_path in _reachable_modules(crate_src):
        text = _read(file_path)
        if text is None:
            result.errors.append(
                OpError(
                    operation=_OPERATION,
                    message=f"crate source unreadable: {file_path}",
                    details={"path": str(file_path), "crate": crate},
                )
            )
            continue
        _scan_region(text, 0, len(text), module_path, emitter)
    return emitter.symbols


def extract(repo_path: Path, release: str) -> OperationResult:
    """Extract the Rust public surface (C ABI + curated libs) from ``repo_path``.

    Returns an :class:`OperationResult` with ``data['symbols']`` and
    ``metadata = {"surface": "rust", "covered": bool, "coverage_detail":
    {"rust:<crate>": bool, ...}}``. A missing curated crate is recorded
    ``False`` in ``coverage_detail`` (so the diff skips it) rather than raising.
    """
    repo_path = Path(repo_path)
    result = OperationResult(
        metadata={"surface": SURFACE, "covered": False, "release": release}
    )
    result.data["symbols"] = []
    coverage_detail: dict[str, bool] = {}
    symbols: list[dict[str, Any]] = []

    if not repo_path.is_dir():
        result.errors.append(
            OpError(
                operation=_OPERATION,
                message=f"repo path not found: {repo_path}",
                details={"repo_path": str(repo_path), "release": release},
            )
        )
        result.metadata["coverage_detail"] = coverage_detail
        return result

    # C ABI slice (high confidence).
    c_crate = rust_crate_for(repo_path / _C_ABI_DIR / "src" / "lib.rs", repo_path)
    c_emitter = _Emitter(c_crate, provenance="c_abi", confidence="high")
    if _extract_c_abi(repo_path, c_crate, c_emitter, result):
        coverage_detail[f"rust:{c_crate}"] = True
        symbols.extend(c_emitter.symbols)
    else:
        coverage_detail[f"rust:{c_crate}"] = False

    # Curated internal libs (medium confidence).
    for lib_dir in CURATED_LIB_DIRS:
        if lib_dir in _EXCLUDED_LIB_DIRS:
            continue
        crate = f"dynamo-{lib_dir}"
        crate_src = repo_path / "lib" / lib_dir / "src"
        crate_symbols = scan_crate_src(crate_src, crate, "lib", "medium", result)
        if crate_symbols is None:
            coverage_detail[f"rust:{crate}"] = False
            continue
        coverage_detail[f"rust:{crate}"] = True
        symbols.extend(crate_symbols)

    result.data["symbols"] = symbols
    result.metadata["covered"] = any(coverage_detail.values())
    result.metadata["coverage_detail"] = coverage_detail
    result.metadata["symbol_count"] = len(symbols)
    return result
