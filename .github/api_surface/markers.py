# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native deprecation marker scanner.

The PRIMARY deprecation signal: Rust ``#[deprecated]`` and Python
``@deprecated`` / ``warnings.warn(DeprecationWarning)``. ``scan_markers``
walks the repository tree, finds these markers in ``*.rs`` / ``*.py`` /
``*.pyi`` sources, and emits one :class:`SurfaceChange` per marker
(``change_type="deprecated_api"``, ``source="marker"``,
``confidence="high"``). The ledger merge consumes these as the highest-
authority deprecation signal -- a marker change always wins over a diff- or
label-sourced entry for the same id.

Symbol id heuristics (best-effort, never lossy):

- Rust: ``rust:<crate>::<module::path>::<name>`` via the shared
  :func:`~api_surface.extractors.rust.rust_id` helper, so
  a marker and an extracted symbol address the same id. The crate and the
  file-level module path come from
  :func:`~api_surface.extractors.rust.rust_crate_for` /
  :func:`~api_surface.extractors.rust.rust_module_path_for`
  (``lib/runtime/src/config.rs`` -> crate ``dynamo-runtime``, module
  ``config``). The scanner sees only the file-level module, not inline ``mod``
  nesting or the enclosing ``impl`` type; the ledger reconciles any residual
  mismatch by ``(crate, trailing item name)``. ``<name>`` is the identifier on
  the deprecated item line (``fn``, ``struct``, ``enum``, ``trait``, ``type``,
  ``const``, ``static``, ``mod``, ``union``, or struct field).
- Python: ``python:<module>.<qualname>``. Module is derived by walking up
  from the file until ``__init__.py`` is no longer present (the parent of
  the topmost ``__init__.py``-bearing directory is the package root);
  top-level scripts with no package fall back to the file stem. Qualname is
  built from the enclosing class/function context. For
  ``warnings.warn(..., DeprecationWarning)`` calls outside any function or
  class, the qualname is empty so the id collapses to ``python:<module>``.

The scan is bounded by design: only text source files are read, common
build/vendor directories (``target``, ``node_modules``, ``__pycache__``,
``.venv`` ...) are pruned in-place during the walk, files larger than
``_MAX_FILE_BYTES`` are skipped, and read errors append :class:`OpError`
rather than raising. Python files that fail to parse are silently skipped --
a real repo will always contain some intentionally-broken test fixtures, and
emitting an OpError per parse miss would drown the legitimate signal.
"""

from __future__ import annotations

import ast
import os
import re
from collections.abc import Iterator
from pathlib import Path

from api_surface.extractors.rust import rust_crate_for, rust_id, rust_module_path_for
from api_surface.models import SurfaceChange
from api_surface.results import OperationResult, OpError

SURFACE = "marker"

_OPERATION = "api_surface.scan.markers"

# Per-file size cap. Real source files comfortably fit; the cap exists to keep
# accidentally-checked-in megabyte JSON / pyi blobs from blowing up the scan.
_MAX_FILE_BYTES = 1_000_000

# Directory names pruned from the walk. Build outputs, virtualenvs, caches,
# and vendored deps -- pruning happens in place on the os.walk dirnames list.
_SKIP_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".idea",
        ".vscode",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        "node_modules",
        "target",  # Rust build output
        "build",
        "dist",
        "vendor",
    }
)

# Matches the literal ``#[deprecated`` token. The ``\b`` rules out
# ``#[allow(deprecated)]`` and similar -- we want an attribute named
# ``deprecated``, not a reference to one.
_RUST_DEPRECATED_RE = re.compile(r"#\[\s*deprecated\b")

# Captures ``note = "..."`` from inside a ``#[deprecated(...)]`` attribute,
# including raw-string forms (``r"..."`` / ``r#"..."#``). DOTALL so a note
# string may span continuation lines.
_RUST_NOTE_RE = re.compile(
    r'note\s*=\s*r?#?"((?:[^"\\]|\\.)*)"#?',
    re.DOTALL,
)

# Item line: optional visibility / async / unsafe / default modifiers, then
# one of the item-introducing keywords, then the identifier.
_RUST_ITEM_RE = re.compile(
    r"""
    ^\s*
    (?:pub(?:\([^)]*\))?\s+)?
    (?:default\s+)?
    (?:async\s+)?
    (?:unsafe\s+)?
    (?:fn|struct|enum|trait|type|const|static|mod|union)
    \s+(\w+)
    """,
    re.VERBOSE,
)

# Struct-field fallback: optional ``pub`` + identifier + colon + non-space.
# Used only when the item regex fails on the line immediately following the
# attribute (so it never triggers on random ``let`` bindings or expressions).
_RUST_FIELD_RE = re.compile(
    r"""
    ^\s*
    (?:pub(?:\([^)]*\))?\s+)?
    (\w+)
    \s*:\s*\S
    """,
    re.VERBOSE,
)

# Source extensions the scanner reads. Each is bounded by ``_MAX_FILE_BYTES``
# and walked with ``os.walk`` once -- not file-suffix iteration -- so a single
# pass picks up both languages.
_SOURCE_SUFFIXES = (".rs", ".py", ".pyi")


def scan_markers(repo_path: Path, release: str) -> OperationResult:
    """Scan native deprecation markers across ``repo_path``.

    Walks the tree once, collecting one :class:`SurfaceChange` per marker.
    The ``release`` argument is recorded in metadata for traceability but
    does not gate the scan -- markers reflect whatever is on disk at the
    given ref.

    Returns an :class:`OperationResult` with::

        data["changes"]      -> list[dict] of SurfaceChange.to_dict()
        metadata["surface"]  -> "marker"
        metadata["covered"]  -> True normally; False only when ``repo_path``
                                itself is missing or not a directory

    Read failures (stat / read / unreadable file) append :class:`OpError`
    rather than raising. ``repo_path`` itself missing is the only condition
    that flips ``covered`` to False -- a tree that exists but happens to
    contain zero markers is a legitimate covered-empty result.
    """
    result = OperationResult(
        metadata={"surface": SURFACE, "covered": False, "release": release}
    )
    result.data["changes"] = []

    if not repo_path.is_dir():
        result.errors.append(
            OpError(
                operation=_OPERATION,
                message=f"Repo path not found: {repo_path}",
                details={"repo_path": str(repo_path), "release": release},
            )
        )
        return result

    # Dedupe by id: one symbol can carry two markers (e.g. a function with both
    # @deprecated and an inner warnings.warn(DeprecationWarning)). Keep a single
    # change per id, preferring whichever carries a real note over the synthetic
    # "Deprecated: <name>" fallback.
    by_id: dict[str, SurfaceChange] = {}
    ids_with_note: set[str] = set()
    for src_path in _iter_source_files(repo_path):
        text = _read_text(src_path, result)
        if text is None:
            continue
        if src_path.suffix == ".rs":
            crate = rust_crate_for(src_path, repo_path)
            module_path = rust_module_path_for(src_path, repo_path)
            for name, note in _scan_rust_text(text):
                _record_change(
                    by_id,
                    ids_with_note,
                    _rust_change(crate, module_path, name, note),
                    bool(note),
                )
        else:
            module = _python_module_for(src_path, repo_path)
            for qualname, note in _scan_python_text(text, src_path):
                _record_change(
                    by_id,
                    ids_with_note,
                    _python_change(module, qualname, note),
                    bool(note),
                )

    changes = sorted(by_id.values(), key=lambda c: (c.surface, c.id, c.summary))
    result.data["changes"] = [c.to_dict() for c in changes]
    result.metadata["covered"] = True
    return result


def _record_change(
    by_id: dict[str, SurfaceChange],
    ids_with_note: set[str],
    change: SurfaceChange,
    has_note: bool,
) -> None:
    """Insert ``change`` keyed by id, preferring an entry that carries a note.

    First write for an id wins; a later write only replaces it when the new
    change has a real note and the stored one did not. This collapses the
    ``@deprecated`` + inner ``warnings.warn`` double-emit into one change.
    """
    if change.id not in by_id:
        by_id[change.id] = change
        if has_note:
            ids_with_note.add(change.id)
        return
    if has_note and change.id not in ids_with_note:
        by_id[change.id] = change
        ids_with_note.add(change.id)


# ---------------------------------------------------------------------------
# File walk + read
# ---------------------------------------------------------------------------


def _iter_source_files(root: Path) -> Iterator[Path]:
    """Yield Rust + Python source files under ``root``, pruning build dirs."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
        for fn in sorted(filenames):
            if fn.endswith(_SOURCE_SUFFIXES):
                yield Path(dirpath) / fn


def _read_text(path: Path, result: OperationResult) -> str | None:
    """Read ``path`` as text, recording an OpError on failure or oversize."""
    try:
        size = path.stat().st_size
    except OSError as exc:
        result.errors.append(
            OpError(
                operation=_OPERATION,
                message=f"stat failed: {path}",
                details={"path": str(path), "error": str(exc)},
            )
        )
        return None
    if size > _MAX_FILE_BYTES:
        return None
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        result.errors.append(
            OpError(
                operation=_OPERATION,
                message=f"read failed: {path}",
                details={"path": str(path), "error": str(exc)},
            )
        )
        return None


# ---------------------------------------------------------------------------
# Rust marker scanning
# ---------------------------------------------------------------------------


def _scan_rust_text(text: str) -> list[tuple[str, str]]:
    """Return ``[(item_name, note), ...]`` for every ``#[deprecated]``.

    The deprecated item is identified by the next item or struct-field line
    after the attribute, skipping blank lines, comments, and other attribute
    spans (``#[derive(...)]``, ``#[builder(...)]``, ``#[cfg(...)]``...).
    Missing notes (the bare ``#[deprecated]`` form) yield ``""``.
    """
    lines = text.splitlines()
    out: list[tuple[str, str]] = []
    n = len(lines)
    i = 0
    while i < n:
        if not _RUST_DEPRECATED_RE.search(lines[i]):
            i += 1
            continue
        attr_text, end_idx = _collect_rust_attr(lines, i)
        note = _extract_rust_note(attr_text)
        target_idx = _next_rust_item_line(lines, end_idx + 1)
        if target_idx is not None:
            name = _extract_rust_item_name(lines[target_idx])
            if name:
                out.append((name, note))
        i = end_idx + 1
    return out


def _collect_rust_attr(lines: list[str], start: int) -> tuple[str, int]:
    """Concatenate the attribute starting at ``lines[start]``.

    Tracks ``[``/``]`` bracket depth across continuation lines until the
    attribute closes. Returns the joined text plus the inclusive end-line
    index. The bracket count is naive about string-literal contents, which
    is fine for real-world ``note=...`` text.
    """
    parts: list[str] = []
    depth = 0
    seen_open = False
    for k in range(start, len(lines)):
        line = lines[k]
        parts.append(line)
        for ch in line:
            if ch == "[":
                depth += 1
                seen_open = True
            elif ch == "]":
                depth -= 1
        if seen_open and depth <= 0:
            return "\n".join(parts), k
    return "\n".join(parts), len(lines) - 1


def _extract_rust_note(attr_text: str) -> str:
    """Return the note value from a ``#[deprecated(...)]`` attribute, or ``""``."""
    match = _RUST_NOTE_RE.search(attr_text)
    if not match:
        return ""
    return _unescape_rust_string(match.group(1))


# Single-character Rust escape sequences the scanner unescapes (anything else
# is kept verbatim). Line continuation (``\<newline><whitespace>*``) is
# handled inline below because it consumes a variable amount of input.
_RUST_SIMPLE_ESCAPES = {
    "n": "\n",
    "t": "\t",
    "r": "\r",
    '"': '"',
    "'": "'",
    "0": "\0",
    "\\": "\\",
}


def _unescape_rust_string(value: str) -> str:
    """Best-effort unescape of common Rust string escapes.

    Handles ``\\"``, ``\\\\``, ``\\n``, ``\\t``, ``\\r``, ``\\'``, ``\\0`` and
    line continuation: a backslash immediately before a newline collapses
    that backslash, the newline, and any leading whitespace on the next line.
    Real ``#[deprecated(note = "...")]`` notes in the dynamo tree use this
    form to wrap long sentences.
    """
    out: list[str] = []
    i = 0
    n = len(value)
    while i < n:
        c = value[i]
        if c == "\\" and i + 1 < n:
            nxt = value[i + 1]
            if nxt == "\n":
                i += 2
                while i < n and value[i] in " \t":
                    i += 1
                continue
            mapped = _RUST_SIMPLE_ESCAPES.get(nxt)
            if mapped is not None:
                out.append(mapped)
                i += 2
                continue
        out.append(c)
        i += 1
    return "".join(out)


def _next_rust_item_line(lines: list[str], start: int) -> int | None:
    """Index of the next non-attribute, non-comment, non-blank line."""
    j = start
    n = len(lines)
    while j < n:
        stripped = lines[j].strip()
        if not stripped or stripped.startswith("//") or stripped.startswith("/*"):
            j += 1
            continue
        if stripped.startswith("#["):
            _, end_idx = _collect_rust_attr(lines, j)
            j = end_idx + 1
            continue
        return j
    return None


def _extract_rust_item_name(line: str) -> str:
    """Return the identifier from an item / struct-field line, or ``""``."""
    item = _RUST_ITEM_RE.match(line)
    if item:
        return item.group(1)
    field = _RUST_FIELD_RE.match(line)
    if field:
        return field.group(1)
    return ""


def _rust_change(crate: str, module_path: str, name: str, note: str) -> SurfaceChange:
    """Build a SurfaceChange for one Rust deprecation marker.

    The id is assembled via the shared :func:`rust_id` helper from the crate,
    the file-level module path, and the item name, so a marker and the rust
    extractor address the same symbol by the same id.
    """
    sid = rust_id(crate, module_path, name)
    summary = note or f"Deprecated: {name}"
    return SurfaceChange(
        change_type="deprecated_api",
        id=sid,
        surface="rust",
        summary=summary,
        confidence="high",
        source="marker",
    )


# ---------------------------------------------------------------------------
# Python marker scanning
# ---------------------------------------------------------------------------


def _scan_python_text(text: str, path: Path) -> list[tuple[str, str]]:
    """Return ``[(qualname, note), ...]`` for every Python deprecation marker.

    A SyntaxError in the source returns an empty list rather than raising:
    real repos always contain intentionally-broken test fixtures, and the
    scan should not pass that noise back to the caller.
    """
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        return []
    out: list[tuple[str, str]] = []
    _walk_python(tree, [], out)
    return out


def _walk_python(node: ast.AST, stack: list[str], out: list[tuple[str, str]]) -> None:
    """Recursive AST walker tracking the enclosing class/function stack.

    For function/class definitions, decorators are evaluated for deprecation
    once and then the body is recursed with the new qualname pushed onto the
    stack -- so a ``warnings.warn(...)`` inside a method attributes to that
    method's qualname, not the enclosing class. Calls inside default values,
    type annotations, and decorator expressions are NOT walked here; that is
    intentional (those positions cannot host a meaningful deprecation
    signal).
    """
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        qualname = ".".join([*stack, node.name])
        for dec in node.decorator_list:
            note = _decorator_dep_note(dec)
            if note is not None:
                out.append((qualname, note))
        for stmt in node.body:
            _walk_python(stmt, [*stack, node.name], out)
        return
    if isinstance(node, ast.Call) and _is_dep_warn_call(node):
        out.append((".".join(stack), _extract_warn_message(node)))
        return
    for child in ast.iter_child_nodes(node):
        _walk_python(child, stack, out)


def _decorator_dep_note(dec: ast.expr) -> str | None:
    """Return the deprecation note when ``dec`` is ``@deprecated`` (any form).

    Recognized: ``@deprecated``, ``@deprecated(msg)``,
    ``@deprecated(message=msg)``, plus the same with any module-attribute
    prefix (``@typing_extensions.deprecated``, ``@warnings.deprecated``...).
    Returns ``None`` if the decorator isn't a deprecation; ``""`` if it is
    but has no extractable string message.
    """
    if _decorator_base_name(dec) != "deprecated":
        return None
    if not isinstance(dec, ast.Call):
        return ""
    for arg in dec.args:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return arg.value
    for kw in dec.keywords:
        if (
            kw.arg in {"message", "reason", "note"}
            and isinstance(kw.value, ast.Constant)
            and isinstance(kw.value.value, str)
        ):
            return kw.value.value
    return ""


def _decorator_base_name(dec: ast.expr) -> str:
    """Rightmost identifier of a decorator expression (``Name``/``Attribute``)."""
    if isinstance(dec, ast.Call):
        return _decorator_base_name(dec.func)
    if isinstance(dec, ast.Name):
        return dec.id
    if isinstance(dec, ast.Attribute):
        return dec.attr
    return ""


def _is_dep_warn_call(call: ast.Call) -> bool:
    """True iff ``call`` is ``warnings.warn(..., DeprecationWarning)``.

    Conservative match: only ``warnings.warn`` (not ``from warnings import
    warn``) so the scanner never misattributes some other ``warn(...)`` as
    a deprecation. ``DeprecationWarning`` may be passed positionally (the
    second arg) or via ``category=...``.
    """
    func = call.func
    if not (
        isinstance(func, ast.Attribute)
        and func.attr == "warn"
        and isinstance(func.value, ast.Name)
        and func.value.id == "warnings"
    ):
        return False
    if len(call.args) >= 2 and _is_deprecation_warning_node(call.args[1]):
        return True
    return any(
        kw.arg == "category" and _is_deprecation_warning_node(kw.value)
        for kw in call.keywords
    )


def _is_deprecation_warning_node(node: ast.expr) -> bool:
    """True iff ``node`` references ``DeprecationWarning`` (any qualifier)."""
    if isinstance(node, ast.Name) and node.id == "DeprecationWarning":
        return True
    return isinstance(node, ast.Attribute) and node.attr == "DeprecationWarning"


def _extract_warn_message(call: ast.Call) -> str:
    """Extract the message string passed to ``warnings.warn(...)``."""
    if (
        call.args
        and isinstance(call.args[0], ast.Constant)
        and isinstance(call.args[0].value, str)
    ):
        return call.args[0].value
    return ""


def _python_module_for(file_path: Path, repo_path: Path) -> str:
    """Best-effort dotted module name for ``file_path``.

    Walks up parent directories while ``__init__.py`` is present (the parent
    of the topmost ``__init__.py``-bearing directory becomes the package
    root). For ``__init__.py`` itself the module is the package path; for
    other files the file stem is appended. Top-level scripts with no
    package fall back to the bare stem.
    """
    abs_file = file_path.resolve()
    abs_root = repo_path.resolve()
    parts: list[str] = []
    parent = abs_file.parent
    while (parent / "__init__.py").is_file():
        parts.append(parent.name)
        if parent == abs_root or parent.parent == parent:
            break
        parent = parent.parent
    parts.reverse()
    if abs_file.stem == "__init__":
        return ".".join(parts)
    return ".".join([*parts, abs_file.stem])


def _python_change(module: str, qualname: str, note: str) -> SurfaceChange:
    """Build a SurfaceChange for one Python deprecation marker."""
    sid = f"python:{module}.{qualname}" if qualname else f"python:{module}"
    summary = note or f"Deprecated: {sid.split(':', 1)[1]}"
    return SurfaceChange(
        change_type="deprecated_api",
        id=sid,
        surface="python",
        summary=summary,
        confidence="high",
        source="marker",
    )
