# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python public-surface extractor for the in-tree ``components/`` packages.

The companion :mod:`python_pyi` extractor covers the three hand-written
``.pyi`` binding stubs. This module covers the *real* ``.py`` public API that
ships in the ``dynamo`` package under ``components/src`` -- the backend
handlers (vLLM / SGLang / TRT-LLM), frontend, planner, router, and profiler
code that users import directly. Both emit into the same ``"python"`` surface;
ids never collide because the stub modules (``dynamo._core``,
``dynamo.prometheus_metrics``, ``kvbm._core``) are all underscore-private
module paths this scanner skips.

Everything is text-mode AST parsing (stdlib :mod:`ast`) -- no imports, no walk
of an installed package -- so the extractor stays import-pure and offline. The
signature/visibility rendering helpers are shared verbatim with
:mod:`python_pyi` so a symbol's id and signature are byte-identical whether it
came from a stub or from source.

Module discovery (offline):

- Anchor on every ``components/**/src/dynamo`` directory (the src-layout
  package root; stable across every release ref surveyed). The dotted module
  name is the path relative to ``src`` with ``/`` -> ``.`` and ``.py``
  stripped; ``__init__.py`` maps to its package.
- A module is private (skipped) when any path segment below ``src`` is
  underscore-prefixed (``_internal``, ``_utils`` ...) or is a non-API tree
  (``tests``, ``examples``, ``benchmarks``). ``__init__.py`` is the only dunder
  file kept -- it *is* the importable package; ``__main__.py`` and other dunder
  / ``conftest`` / ``test_*`` files are scaffolding, not import API.

Symbol scoping differs from the stub extractor in exactly one place. A ``.pyi``
stub only lists genuine public constants, so every top-level assignment is
public. A real module is full of private module state (``logger = ...``,
``app = FastAPI()``, compiled regexes), so a bare lowercase top-level
assignment is almost never API: it is captured only when the author published
it via ``__all__`` or it is an ``ALL_CAPS`` constant. Functions, classes (and
their public members), and explicitly typed (``AnnAssign``) attributes follow
the same public/private rules as :mod:`python_pyi`.

Failures degrade: an unparseable file appends an :class:`OpError` and is
skipped; a missing ``components/`` tree records the gap in ``coverage_detail``
and returns an empty, well-formed result without aborting. Coverage is
reported only through ``metadata['coverage_detail']`` (key
``python:components``) -- never through ``metadata['surface']`` / ``covered``
-- so it supplements rather than clobbers the stub extractor's ``python``
coverage bit, supplementing the curated Python-stub surface.
"""

from __future__ import annotations

import ast
from pathlib import Path

from api_surface.extractors.python_pyi import (
    SURFACE,
    _coalesce_callable_overloads,
    _decorator_names,
    _extract_class,
    _format_annotation,
    _format_signature,
    _is_public_top,
    _read_all,
)
from api_surface.models import SurfaceSymbol
from api_surface.results import OperationResult, OpError

# Path segments (below the ``dynamo`` package root) that are never public API.
_EXCLUDE_SEGMENTS = frozenset({"tests", "test", "examples", "benchmarks"})


def _declared_public(name: str, all_names: set[str] | None) -> bool:
    """Positive public-API signal for a ``.py`` symbol: ``__all__`` membership.

    A symbol is *declared* public only when its module lists it in ``__all__``.
    Non-underscore names in a module with no ``__all__`` are still emitted (the
    surface stays fully tracked for diffing) but carry no stability promise, so
    the stability layer defaults them to ``experimental``. This is what stops an
    internal-utils refactor (``planner.utils`` and friends, none of which
    declare ``__all__``) from registering as a wave of broken stability
    promises.
    """
    return all_names is not None and name in all_names


# Largest file we will parse; guards against a pathological generated module.
_MAX_BYTES = 2 * 1024 * 1024


def _iter_package_roots(repo_path: Path) -> list[Path]:
    """Return every ``components/**/src/dynamo`` package root under ``repo_path``.

    Covers both the flat ``components/src/dynamo`` layout and any future
    per-component ``components/<name>/src/dynamo`` split. Results are sorted and
    de-duplicated for deterministic output.
    """
    components = repo_path / "components"
    if not components.is_dir():
        return []
    roots = {p for p in components.glob("**/src/dynamo") if p.is_dir()}
    return sorted(roots)


def _module_name(src_dir: Path, py_file: Path) -> str | None:
    """Dotted module name for ``py_file``, or ``None`` when it is non-public.

    ``src_dir`` is the ``dynamo`` package root; its parent is the ``src``
    anchor. A module is non-public when any segment is underscore-prefixed
    (other than the kept ``__init__``) or lives in an excluded tree, or when
    the file is ``conftest`` / ``test_*`` / ``*_test`` scaffolding.
    """
    rel = py_file.relative_to(src_dir.parent)
    parts = list(rel.parts)
    filename = parts[-1]

    if filename == "__init__.py":
        parts = parts[:-1]
    else:
        stem = filename[:-3] if filename.endswith(".py") else filename
        if stem.startswith("_") or stem == "conftest":
            return None
        if stem.startswith("test_") or stem.endswith("_test"):
            return None
        parts[-1] = stem

    for seg in parts:
        if seg.startswith("_") or seg in _EXCLUDE_SEGMENTS:
            return None
    return ".".join(parts)


def _extract_module(source: str, module: str, symbols: list[SurfaceSymbol]) -> None:
    """Parse one real ``.py`` module and append its public symbols.

    Mirrors :func:`python_pyi._extract_module` but tightens the bare-assignment
    branch (see module docstring) so private module state does not masquerade as
    public API.
    """
    tree = ast.parse(source)
    all_names = _read_all(tree)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if not _is_public_top(node.name, all_names):
                continue
            symbols.append(
                SurfaceSymbol(
                    surface=SURFACE,
                    kind="function",
                    id=f"python:{module}.{node.name}",
                    signature=_format_signature(node),
                    metadata={
                        "module": module,
                        "declared_public": _declared_public(node.name, all_names),
                        "overload": "overload" in _decorator_names(node.decorator_list),
                    },
                )
            )
        elif isinstance(node, ast.ClassDef):
            if not _is_public_top(node.name, all_names):
                continue
            _extract_class(
                node,
                module,
                symbols,
                declared_public=_declared_public(node.name, all_names),
            )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
            if name == "__all__" or not _is_public_top(name, all_names):
                continue
            symbols.append(
                SurfaceSymbol(
                    surface=SURFACE,
                    kind="attribute",
                    id=f"python:{module}.{name}",
                    signature=_format_annotation(node.annotation),
                    metadata={
                        "module": module,
                        "declared_public": _declared_public(name, all_names),
                    },
                )
            )
        elif isinstance(node, ast.Assign):
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                continue
            name = node.targets[0].id
            if name == "__all__":
                continue
            published = all_names is not None and name in all_names
            is_const = name.isupper() and not name.startswith("_")
            if not (published or is_const):
                continue
            symbols.append(
                SurfaceSymbol(
                    surface=SURFACE,
                    kind="attribute",
                    id=f"python:{module}.{name}",
                    signature="",
                    metadata={"module": module, "declared_public": published},
                )
            )


def extract(repo_path: Path, release: str) -> OperationResult:
    """Extract the Python public surface from the ``components/`` packages.

    Args:
        repo_path: Root of a checked-out source tree.
        release: Release version the extraction is anchored to (recorded in
            ``metadata['release']``; not used to gate parsing).

    Returns:
        An :class:`OperationResult` whose ``data['symbols']`` is a sorted list
        of :meth:`SurfaceSymbol.to_dict` dicts on the ``"python"`` surface.
        ``metadata['coverage_detail']['python:components']`` is ``True`` iff at
        least one module parsed; per-file parse failures append an
        :class:`OpError` (never raised) and are skipped.
    """
    result = OperationResult(metadata={"release": release})
    symbols: list[SurfaceSymbol] = []
    roots = _iter_package_roots(repo_path)
    parsed_any = False

    if not roots:
        result.errors.append(
            OpError(
                operation="extract_python_components",
                message="no components/**/src/dynamo package root found",
                details={"repo_path": str(repo_path)},
            )
        )
        result.metadata["coverage_detail"] = {"python:components": False}
        result.data["symbols"] = []
        result.metadata["symbol_count"] = 0
        return result

    for src_dir in roots:
        for py_file in sorted(src_dir.rglob("*.py")):
            module = _module_name(src_dir, py_file)
            if module is None:
                continue
            try:
                if py_file.stat().st_size > _MAX_BYTES:
                    continue
                source = py_file.read_text(encoding="utf-8")
                _extract_module(source, module, symbols)
                parsed_any = True
            except SyntaxError as exc:
                result.errors.append(
                    OpError(
                        operation="extract_python_components",
                        message=f"parse failed for {module}: {exc.msg}",
                        details={
                            "path": str(py_file),
                            "module": module,
                            "line": exc.lineno,
                        },
                    )
                )
            except OSError as exc:
                result.errors.append(
                    OpError(
                        operation="extract_python_components",
                        message=f"read failed for {module}: {exc}",
                        details={"path": str(py_file), "module": module},
                    )
                )

    symbols = _coalesce_callable_overloads(symbols)
    symbols.sort(key=lambda s: s.id)
    result.data["symbols"] = [s.to_dict() for s in symbols]
    result.metadata["coverage_detail"] = {"python:components": parsed_any}
    result.metadata["symbol_count"] = len(symbols)
    return result
