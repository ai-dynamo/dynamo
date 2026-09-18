# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python public-surface extractor (slot B2).

Statically parses a small allowlist of ``.pyi`` stubs and emits one
:class:`SurfaceSymbol` per public function / method / class / attribute. No
runtime code is imported and no live tree is walked: the entire surface comes
from text-mode AST parsing of stub files, so the engine stays import-pure and
offline.

The stdlib :mod:`ast` module is the right tool here. ``griffe`` is ideal for a
full installed package but is awkward to point at a bare ``.pyi`` stub
divorced from its parent package layout, so we stay in the standard library.

Why each design choice exists:

- Symbol ids follow the frozen grammar in
  :mod:`api_surface.models` -- ``python:<module>.<qualname>``
  -- so the diff engine matches symbols across releases by string equality.
- Signatures are rendered through :func:`ast.unparse` and then run through
  :func:`canonicalize_signature` so cosmetic whitespace cannot register as a
  signature change in the diff engine.
- ``__all__`` is treated as *additive*, not restrictive. For surface tracking
  a non-underscore top-level definition is public regardless of ``__all__`` --
  ``module.DistributedRuntime`` is reachable even when ``__all__`` omits it
  (``__all__`` only governs ``from module import *``). So a name is public when
  it is non-underscore, OR it is explicitly listed in ``__all__`` (which lets a
  stub deliberately export an underscore-prefixed name). This avoids the failure
  mode where an incomplete ``__all__`` (e.g. Dynamo's v1.0.0 ``_core.pyi`` lists
  7 names while defining 35 public classes) under-reports the surface and makes
  the next release -- which dropped ``__all__`` -- look like it added them.
  Dunder methods on public classes (``__init__``, ``__aiter__``, ``__anext__``)
  are always kept because they encode real construction / iteration shape.
- Missing stubs are NOT a hard failure. The diff engine relies on
  ``metadata['covered']`` to distinguish a coverage gap from a wave of
  removals, so this extractor records the gap, appends an :class:`OpError`,
  and returns an empty (but well-formed) result.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from pathlib import Path

from api_surface.diff import canonicalize_signature
from api_surface.models import SurfaceSymbol
from api_surface.results import OperationResult, OpError

SURFACE = "python"

# Tracked stubs: ``(relative path within repo, module dotted name)``.
# The hand-written ``.pyi`` stubs that ship the public binding API. The
# generated ``plugin_pb2.pyi`` (protobuf) and the ``_internal`` stub are
# intentionally excluded; ``components/`` ships ``.py`` (not stubs) and is
# out of scope for this stub-based extractor.
DEFAULT_STUBS: tuple[tuple[str, str], ...] = (
    ("lib/bindings/python/src/dynamo/_core.pyi", "dynamo._core"),
    (
        "lib/bindings/python/src/dynamo/prometheus_metrics.pyi",
        "dynamo.prometheus_metrics",
    ),
    ("lib/bindings/kvbm/python/kvbm/_core.pyi", "kvbm._core"),
)


def _format_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Render a canonical ``(args) -> return`` string for a callable node."""
    try:
        args_src = ast.unparse(node.args)
    except Exception:
        args_src = ""
    ret_src = ""
    if node.returns is not None:
        try:
            ret_src = " -> " + ast.unparse(node.returns)
        except Exception:
            ret_src = ""
    return canonicalize_signature(f"({args_src}){ret_src}")


def _format_annotation(annotation: ast.expr | None) -> str:
    """Canonical annotation text for an attribute (empty when absent)."""
    if annotation is None:
        return ""
    try:
        return canonicalize_signature(ast.unparse(annotation))
    except Exception:
        return ""


def _decorator_names(decorators: list[ast.expr]) -> set[str]:
    """Plain decorator names (``property``, ``staticmethod``, ...).

    Treats ``@foo``, ``@m.foo``, and ``@foo(...)`` identically -- we only need
    to know whether ``property`` (or its alias) was applied, not its args.
    """
    names: set[str] = set()
    for dec in decorators:
        target = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, ast.Attribute):
            names.add(target.attr)
    return names


def _read_all(tree: ast.Module) -> set[str] | None:
    """Return ``__all__`` membership as a set, or ``None`` when absent.

    Accepts the common literal forms (list / tuple / set of string constants).
    A non-literal ``__all__`` (computed at import time) is ignored -- there is
    no safe static interpretation, so we fall back to underscore filtering.
    """
    for node in tree.body:
        targets: list[ast.expr] = []
        value: ast.expr | None = None
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
            value = node.value
        elif isinstance(node, ast.AnnAssign) and node.target is not None:
            targets = [node.target]
            value = node.value
        if value is None:
            continue
        if not any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
            continue
        if isinstance(value, ast.List | ast.Tuple | ast.Set):
            collected: list[str] = []
            for elt in value.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    collected.append(elt.value)
            return set(collected)
    return None


def _is_public_top(name: str, all_names: set[str] | None) -> bool:
    """Top-level visibility: non-underscore is public; ``__all__`` only adds.

    ``__all__`` is additive (it can expose an underscore-prefixed name), never
    restrictive: a non-underscore top-level definition stays public even when an
    incomplete ``__all__`` omits it, because it remains reachable via attribute
    access. See the module docstring for why this matters across releases.
    """
    if not name.startswith("_"):
        return True
    return all_names is not None and name in all_names


def _is_public_member(name: str) -> bool:
    """Class members keep dunders; single-underscore privates drop."""
    if name.startswith("__") and name.endswith("__") and len(name) > 4:
        return True
    return not name.startswith("_")


def _extract_class(
    cls: ast.ClassDef,
    module: str,
    symbols: list[SurfaceSymbol],
    qualprefix: str = "",
    declared_public: bool = True,
) -> None:
    """Append the class + its public members.

    ``qualprefix`` is the dotted qualname of any enclosing class (empty at
    module top level), so a nested class becomes ``Outer.Inner`` and its
    members ``Outer.Inner.method`` -- nested classes are recursed into rather
    than dropped.

    ``declared_public`` records whether the class carries a positive public-API
    signal (``__all__`` membership for ``.py`` modules; always ``True`` for the
    curated ``.pyi`` stubs). Public members inherit it: a method of an
    undeclared class is itself undeclared, so the stability layer can demote the
    whole class to ``experimental`` rather than promise stability on it.
    """
    qualname = f"{qualprefix}.{cls.name}" if qualprefix else cls.name
    symbols.append(
        SurfaceSymbol(
            surface=SURFACE,
            kind="class",
            id=f"python:{module}.{qualname}",
            signature="",
            metadata={"module": module, "declared_public": declared_public},
        )
    )
    for node in cls.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if not _is_public_member(node.name):
                continue
            decorators = _decorator_names(node.decorator_list)
            if "property" in decorators:
                kind = "attribute"
                sig = _format_annotation(node.returns)
                suffix = ""
            elif "setter" in decorators:
                kind = "property_setter"
                sig = _format_signature(node)
                suffix = "@setter"
            elif "deleter" in decorators:
                kind = "property_deleter"
                sig = _format_signature(node)
                suffix = "@deleter"
            else:
                kind = "method"
                sig = _format_signature(node)
                suffix = ""
            symbols.append(
                SurfaceSymbol(
                    surface=SURFACE,
                    kind=kind,
                    id=f"python:{module}.{qualname}.{node.name}{suffix}",
                    signature=sig,
                    metadata={
                        "module": module,
                        "declared_public": declared_public,
                        "overload": "overload" in decorators,
                    },
                )
            )
        elif isinstance(node, ast.ClassDef):
            if not _is_public_member(node.name):
                continue
            _extract_class(
                node,
                module,
                symbols,
                qualprefix=qualname,
                declared_public=declared_public,
            )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
            if not _is_public_member(name):
                continue
            symbols.append(
                SurfaceSymbol(
                    surface=SURFACE,
                    kind="attribute",
                    id=f"python:{module}.{qualname}.{name}",
                    signature=_format_annotation(node.annotation),
                    metadata={"module": module, "declared_public": declared_public},
                )
            )


def _extract_module(source: str, module: str, symbols: list[SurfaceSymbol]) -> None:
    """Parse one ``.pyi`` source string and append discovered public symbols.

    Every symbol is tagged ``declared_public=True``: a hand-written ``.pyi``
    binding stub *is* the public-API declaration, so its symbols carry the
    stability promise even when the stub's ``__all__`` is incomplete (e.g.
    ``_core.pyi`` lists 7 of 35 public classes). The companion ``.py``
    components extractor, where module discovery is automatic, instead keys the
    flag off ``__all__`` membership.
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
                        "declared_public": True,
                        "overload": "overload" in _decorator_names(node.decorator_list),
                    },
                )
            )
        elif isinstance(node, ast.ClassDef):
            if not _is_public_top(node.name, all_names):
                continue
            _extract_class(node, module, symbols, declared_public=True)
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
                    metadata={"module": module, "declared_public": True},
                )
            )
        elif isinstance(node, ast.Assign):
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                continue
            name = node.targets[0].id
            if name == "__all__" or not _is_public_top(name, all_names):
                continue
            symbols.append(
                SurfaceSymbol(
                    surface=SURFACE,
                    kind="attribute",
                    id=f"python:{module}.{name}",
                    signature="",
                    metadata={"module": module, "declared_public": True},
                )
            )


def _coalesce_callable_overloads(
    symbols: list[SurfaceSymbol],
) -> list[SurfaceSymbol]:
    """Represent one overloaded callable with one stable ID and signature set.

    Property accessors receive distinct IDs in :func:`_extract_class`. True
    ``@overload`` declarations intentionally share an ID, so their canonical
    signatures are joined deterministically instead of violating snapshot ID
    uniqueness.
    """
    grouped: dict[str, list[SurfaceSymbol]] = {}
    for symbol in symbols:
        grouped.setdefault(symbol.id, []).append(symbol)

    coalesced: list[SurfaceSymbol] = []
    for group in grouped.values():
        overload = any(bool(symbol.metadata.get("overload")) for symbol in group)
        callable_group = all(symbol.kind in {"function", "method"} for symbol in group)
        if len(group) == 1 or not (overload and callable_group):
            coalesced.extend(group)
            continue
        representative = group[0]
        signatures = sorted({symbol.signature for symbol in group if symbol.signature})
        representative.signature = " || ".join(signatures)
        representative.metadata = dict(representative.metadata)
        representative.metadata["overload_count"] = len(signatures)
        representative.metadata.pop("overload", None)
        coalesced.append(representative)
    return coalesced


def extract(
    repo_path: Path,
    release: str,
    *,
    stubs: Iterable[tuple[str, str]] | None = None,
) -> OperationResult:
    """Extract the Python public surface from the configured ``.pyi`` stubs.

    Args:
        repo_path: Repository root to resolve stub paths against.
        release: Release version the extraction is anchored to (recorded in
            ``metadata['release']``; not used to gate parsing).
        stubs: Optional override list of ``(relative_path, module_name)``
            pairs. Production callers leave this defaulted to
            :data:`DEFAULT_STUBS`; tests use it to point at synthetic
            fixtures. The positional protocol surface is unchanged so the
            :class:`SurfaceExtractor` contract still holds.

    Returns:
        An :class:`OperationResult` whose ``data['symbols']`` is a sorted list
        of :meth:`SurfaceSymbol.to_dict` dicts. ``metadata['surface']`` is
        always ``"python"``. ``metadata['covered']`` is ``True`` iff at least
        one configured stub was parsed; otherwise ``False`` and an
        :class:`OpError` is appended (never raised).
    """
    result = OperationResult(
        metadata={"surface": SURFACE, "release": release, "covered": False}
    )
    stub_list = list(stubs) if stubs is not None else list(DEFAULT_STUBS)
    symbols: list[SurfaceSymbol] = []
    extracted_any = False

    for rel_path, module in stub_list:
        stub = repo_path / rel_path
        if not stub.exists():
            result.errors.append(
                OpError(
                    operation="extract_python_pyi",
                    message=f"stub not found: {rel_path}",
                    details={"path": str(stub), "module": module},
                )
            )
            continue
        try:
            source = stub.read_text(encoding="utf-8")
            _extract_module(source, module, symbols)
            extracted_any = True
        except SyntaxError as exc:
            result.errors.append(
                OpError(
                    operation="extract_python_pyi",
                    message=f"parse failed for {rel_path}: {exc.msg}",
                    details={"path": str(stub), "module": module, "line": exc.lineno},
                )
            )
        except OSError as exc:
            result.errors.append(
                OpError(
                    operation="extract_python_pyi",
                    message=f"read failed for {rel_path}: {exc}",
                    details={"path": str(stub), "module": module},
                )
            )

    symbols = _coalesce_callable_overloads(symbols)
    symbols.sort(key=lambda s: s.id)
    result.data["symbols"] = [s.to_dict() for s in symbols]
    result.metadata["covered"] = extracted_any
    result.metadata["symbol_count"] = len(symbols)
    return result
