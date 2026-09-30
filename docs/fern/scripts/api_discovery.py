# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static Python API discovery for the Dynamo docs generator.

Owns the curated ``MODULES`` list, the frozen data model (``Module`` /
``Symbol`` / ``Method``), and the griffe-driven discovery pipeline that
turns each curated package into an ordered set of documented symbols. No
Dynamo code is imported at runtime; the source tree is walked statically.

This module is imported by:

* :mod:`api_rendering` -- receives ``Module`` / ``Symbol`` / ``Method``
  values and turns them into TypeScript / MDX text.
* :mod:`gen_python_api` -- the CLI orchestrator, which calls
  :func:`discover_all_modules` to obtain the list once per run.

Tests in ``docs/fern/scripts/tests`` may monkeypatch
:func:`discover_all_modules` on the CLI orchestrator to reuse a session
cache and avoid re-parsing the full source tree.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from griffe import Alias, AliasResolutionError, Class, Function, GriffeLoader, Kind
from griffe import Module as GriffeModule
from griffe import Parameter, ParameterKind, Parser

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent

SEARCH_PATH_PARTS: tuple[tuple[str, ...], ...] = (
    ("lib", "bindings", "python", "src"),
    ("components", "src"),
)

MODULES: tuple[tuple[str, str, str], ...] = (
    (
        "dynamo._core",
        "_core",
        "Rust-backed distributed runtime, KV router, and endpoint bindings.",
    ),
    (
        "dynamo.runtime",
        "runtime",
        "Decorators and re-exports for defining Dynamo workers and endpoints.",
    ),
    (
        "dynamo.llm",
        "llm",
        "High-level LLM primitives for building request pipelines.",
    ),
    (
        "dynamo.frontend",
        "frontend",
        "OpenAI-compatible HTTP frontend, argument parsing, and pre/post-processing.",
    ),
    (
        "dynamo.common",
        "common",
        "Shared configuration groups, storage adapters, and utility helpers.",
    ),
    (
        "dynamo.health_check",
        "health_check",
        "Health-check payload types and environment-driven configuration.",
    ),
    (
        "dynamo.logits_processing",
        "logits_processing",
        "Custom logits processors for LLM token generation.",
    ),
    (
        "dynamo.planner",
        "planner",
        "Scaling connectors, decision types, and Planner configuration.",
    ),
    (
        "dynamo.router",
        "router",
        "Request-router configuration and command-line argument groups.",
    ),
    (
        "dynamo.mocker",
        "mocker",
        "Mock engine used to run Dynamo workflows without a real GPU backend.",
    ),
    (
        "dynamo.nixl_connect",
        "nixl_connect",
        "NIXL RDMA connector primitives for KV cache and tensor transport.",
    ),
)

SKIP_SUBMODULES: frozenset[str] = frozenset({"tests", "proto", "plugins"})

SOURCE_BRANCH = "main"
SOURCE_BASE = f"https://github.com/ai-dynamo/dynamo/blob/{SOURCE_BRANCH}"

# Dynamo mixes docstring styles -- most packages are Google-style while
# ``dynamo.nixl_connect`` is NumPy-style -- so the style is detected per
# docstring rather than pinned. Griffe logs a warning for every parameter it
# cannot reconcile with the signature, which is noise for a docs build, so
# warnings are disabled for whichever style wins detection.
DOCSTRING_PARSER_OPTIONS: dict[Parser, dict[str, bool]] = {
    parser: {"warnings": False}
    for parser in (Parser.google, Parser.numpy, Parser.sphinx)
}

# Griffe section kinds carrying named entries the renderer turns into terms.
_TERM_SECTION_KINDS: frozenset[str] = frozenset(
    {"parameters", "other parameters", "attributes", "returns", "yields", "raises"}
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DocTerm:
    """One named entry inside a docstring section.

    Covers parameters, attributes, returns, and raises alike: griffe models
    each as a name and/or annotation plus a description, and the renderer
    only needs to know which of the two to use as the label.
    """

    name: str
    annotation: str
    description: str


@dataclass(frozen=True)
class DocSection:
    """One parsed docstring section, normalized off griffe's own types.

    A single flat shape beats a class per section kind: the renderer just
    dispatches on ``kind`` and reads whichever field that kind populates.

    ``kind`` is griffe's section kind (``text``, ``parameters``,
    ``attributes``, ``returns``, ``yields``, ``raises``, ``examples``,
    ``admonition``). ``label`` carries griffe's normalized admonition kind
    (``note``, ``warning``) and is empty otherwise. ``text`` holds prose,
    ``terms`` holds named entries, and ``blocks`` holds the ordered
    ``(kind, text)`` pairs of an examples section, where an ``examples`` kind
    is a code sample and a ``text`` kind is prose between samples.
    """

    kind: str
    label: str = ""
    text: str = ""
    terms: tuple[DocTerm, ...] = ()
    blocks: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Method:
    """One public method on a class, as surfaced in the expanded body.

    ``docs`` carries the same parsed sections as :attr:`Symbol.docs`, so a
    method's parameters, returns, raises, and admonitions render through the
    one path rather than being flattened into :attr:`summary`.
    """

    name: str
    signature: str
    summary: str
    source_path: str
    source_line: int
    source_href: str
    docs: tuple[DocSection, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Symbol:
    """One documented public symbol on a module (class or function)."""

    name: str
    kind: str  # "class" or "function"
    qualname: str  # canonical import path, e.g. "dynamo._core.Client"
    import_path: str  # public import path, preserving aliases
    summary: str
    signature: str  # signature text; empty for classes without __init__
    source_path: str  # repo-relative POSIX path
    source_line: int
    source_href: str
    methods: tuple[Method, ...] = field(default_factory=tuple)
    docs: tuple[DocSection, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Module:
    """One documented Python module, with its ordered public symbols."""

    name: str
    slug: str
    summary: str
    source_path: str  # repo-relative POSIX path to the module root
    source_href: str
    symbols: tuple[Symbol, ...]


# ---------------------------------------------------------------------------
# Griffe discovery (static; no Dynamo import)
# ---------------------------------------------------------------------------


def build_loader(repo_root: Path | None = None) -> GriffeLoader:
    """Griffe loader wired to Dynamo's two Python source roots."""
    root = repo_root or REPO_ROOT
    search_paths = [str(root.joinpath(*parts)) for parts in SEARCH_PATH_PARTS]
    return GriffeLoader(search_paths=search_paths)


def discover_module(loader: GriffeLoader, spec: tuple[str, str, str]) -> Module:
    """Load one module via griffe and return its ordered public symbols."""
    name, slug, summary = spec
    griffe_mod = loader.load(name)
    if not isinstance(griffe_mod, GriffeModule):
        raise TypeError(f"expected Module for {name}, got {type(griffe_mod)}")
    # ``qualname`` breaks ties between same-named symbols re-exported from
    # different submodules (``dynamo.frontend.prepost`` vs
    # ``dynamo.frontend.sglang_prepost`` both expose
    # ``build_tool_call_guided_decoding``). Without it the pair falls back to
    # griffe's member iteration order, which reshuffles the pages between runs
    # and fails --check with no underlying source change.
    symbols = tuple(
        sorted(
            _collect_symbols(name, griffe_mod),
            key=lambda s: (s.kind, s.name, s.qualname),
        )
    )
    return Module(
        name=name,
        slug=slug,
        summary=summary,
        source_path=_relpath(griffe_mod.filepath),
        source_href=_source_href(_relpath(griffe_mod.filepath), 0),
        symbols=symbols,
    )


def discover_all_modules(loader: GriffeLoader | None = None) -> list[Module]:
    """Discover every curated module in :data:`MODULES` in one pass.

    Tests monkeypatch this function on the CLI orchestrator module so a
    session-scoped fixture can serve the same discovery result to every
    I/O and ``--check`` test without re-running griffe per test.
    """
    active_loader = loader or build_loader()
    return [discover_module(active_loader, spec) for spec in MODULES]


def _collect_symbols(pkg_name: str, mod: GriffeModule) -> Iterator[Symbol]:
    """Top-level re-exports first, then symbols defined in direct submodules."""
    seen: set[tuple[str, str]] = set()
    yield from _iter_top_level_symbols(mod, seen)
    yield from _iter_submodule_symbols(pkg_name, mod, seen)


def _iter_top_level_symbols(
    mod: GriffeModule, seen: set[tuple[str, str]]
) -> Iterator[Symbol]:
    """Emit public class/function members declared or re-exported at the top."""
    for name, member in mod.members.items():
        if _is_private_name(name):
            continue
        target = _resolve_dynamo_target(member)
        if target is None:
            continue
        symbol = _build_symbol(
            name,
            target,
            import_path=f"{mod.canonical_path}.{name}",
        )
        key = (symbol.qualname, symbol.name)
        if key in seen:
            continue
        seen.add(key)
        yield symbol


def _iter_submodule_symbols(
    pkg_name: str, mod: GriffeModule, seen: set[tuple[str, str]]
) -> Iterator[Symbol]:
    """Emit symbols DEFINED in each direct submodule (skips SKIP_SUBMODULES)."""
    for name, member in mod.members.items():
        if _is_private_name(name) or name in SKIP_SUBMODULES:
            continue
        target = member if not isinstance(member, Alias) else _try_resolve(member)
        if not isinstance(target, GriffeModule):
            continue
        subcp = str(target.canonical_path)
        if not subcp.startswith(pkg_name + "."):
            continue
        yield from _iter_defined_in(target, subcp, seen)


def _iter_defined_in(
    submodule: GriffeModule, prefix: str, seen: set[tuple[str, str]]
) -> Iterator[Symbol]:
    """Yield class/function members whose canonical_path lives under prefix."""
    for name, member in submodule.members.items():
        if _is_private_name(name):
            continue
        target = _resolve_dynamo_target(member)
        if target is None:
            continue
        canonical = str(target.canonical_path)
        if not (canonical == prefix or canonical.startswith(prefix + ".")):
            continue
        symbol = _build_symbol(name, target, import_path=f"{prefix}.{name}")
        key = (symbol.qualname, symbol.name)
        if key in seen:
            continue
        seen.add(key)
        yield symbol


def _is_private_name(name: str) -> bool:
    """Skip leading-underscore names, except the ``_core`` binding stub."""
    return name.startswith("_") and not name.startswith("_core")


def _try_resolve(member: object) -> object | None:
    """Follow an alias to its target; return None if it cannot be resolved."""
    if not isinstance(member, Alias):
        return member
    try:
        return member.final_target
    except AliasResolutionError:
        return None


def _resolve_dynamo_target(member: object) -> Class | Function | None:
    """Resolve to a Class/Function whose canonical_path lives under dynamo."""
    if isinstance(member, Alias):
        if not str(member.target_path).startswith("dynamo."):
            return None
        target: object | None = _try_resolve(member)
    else:
        target = member
    if not isinstance(target, (Class, Function)):
        return None
    if not str(target.canonical_path).startswith("dynamo."):
        return None
    return target


def _build_symbol(name: str, target: Class | Function, *, import_path: str) -> Symbol:
    """Adapt one resolved griffe class/function into a :class:`Symbol`."""
    kind = "class" if target.kind is Kind.CLASS else "function"
    summary = _first_docstring_line(target)
    signature = (
        _class_signature(target)
        if isinstance(target, Class)
        else _function_signature(target)
    )
    source_path = _relpath(target.filepath)
    source_line = int(target.lineno or 0)
    methods = _iter_public_methods(target) if isinstance(target, Class) else ()
    return Symbol(
        name=name,
        kind=kind,
        qualname=str(target.canonical_path),
        import_path=import_path,
        summary=summary,
        signature=signature,
        source_path=source_path,
        source_line=source_line,
        source_href=_source_href(source_path, source_line),
        methods=tuple(methods),
        docs=_docstring_sections(target),
    )


def _first_docstring_line(target: Class | Function) -> str:
    """First non-empty line of the docstring, or an empty string."""
    doc = getattr(target, "docstring", None)
    if not doc or not doc.value:
        return ""
    for line in doc.value.strip().splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _docstring_sections(target: Class | Function) -> tuple[DocSection, ...]:
    """Every structured docstring section, in authored order.

    The leading text section is kept whole, including the opening sentence
    that :func:`_first_docstring_line` truncates at the first newline. The
    renderer decides how to split lead from body; discarding the remainder
    here would lose the tail of any summary sentence that wraps. Sections
    griffe recognizes but this generator has no placement for are skipped
    rather than dumped as raw text.
    """
    doc = getattr(target, "docstring", None)
    if not doc or not doc.value:
        return ()
    parsed = doc.parse(Parser.auto, per_style_options=DOCSTRING_PARSER_OPTIONS)
    sections = (_convert_section(s.kind.value, s) for s in parsed)
    return tuple(section for section in sections if section is not None)


def _convert_section(kind: str, section: object) -> DocSection | None:
    """Adapt one griffe section, or return None when it has no placement."""
    value = getattr(section, "value", None)
    if kind == "text":
        text = str(value).strip()
        return DocSection(kind="text", text=text) if text else None
    if kind in _TERM_SECTION_KINDS:
        terms = tuple(_doc_term(entry) for entry in value or ())
        normalized = "parameters" if kind == "other parameters" else kind
        return DocSection(kind=normalized, terms=terms) if terms else None
    if kind == "examples":
        blocks = tuple(
            (block_kind.value, str(block_text).strip())
            for block_kind, block_text in value or ()
            if str(block_text).strip()
        )
        return DocSection(kind="examples", blocks=blocks) if blocks else None
    if kind == "admonition":
        contents = str(getattr(value, "contents", "")).strip()
        label = str(getattr(value, "kind", "") or "").strip().lower()
        if not contents:
            return None
        return DocSection(kind="admonition", label=label, text=contents)
    return None


def _doc_term(entry: object) -> DocTerm:
    """One named docstring entry, with griffe expressions flattened to text."""
    annotation = getattr(entry, "annotation", None)
    return DocTerm(
        name=str(getattr(entry, "name", "") or ""),
        annotation="" if annotation is None else str(annotation),
        description=str(getattr(entry, "description", "") or "").strip(),
    )


def _class_signature(cls: Class) -> str:
    """Render the class's constructor signature, if it defines ``__init__``."""
    init = cls.members.get("__init__") if hasattr(cls, "members") else None
    resolved = _try_resolve(init) if init is not None else None
    if isinstance(resolved, Function):
        return _function_signature(resolved).replace("__init__", cls.name, 1)
    return ""


def _function_signature(func: Function) -> str:
    """Render ``name(param: T = default, ...) -> return`` from griffe metadata."""
    parameters = [param for param in func.parameters if param.name != "self"]
    parts: list[str] = []
    has_var_positional = False
    has_keyword_separator = False
    for index, param in enumerate(parameters):
        if (
            param.kind == ParameterKind.keyword_only
            and not has_var_positional
            and not has_keyword_separator
        ):
            parts.append("*")
            has_keyword_separator = True
        parts.append(_render_parameter(param))
        if param.kind == ParameterKind.var_positional:
            has_var_positional = True
        if param.kind == ParameterKind.positional_only:
            next_kind = (
                parameters[index + 1].kind if index + 1 < len(parameters) else None
            )
            if next_kind != ParameterKind.positional_only:
                parts.append("/")
    returns = f" -> {func.returns}" if func.returns else ""
    return f"{func.name}({', '.join(parts)}){returns}"


def _render_parameter(param: Parameter) -> str:
    """Format one griffe parameter as ``name: annotation = default``."""
    prefix = ""
    if param.kind == ParameterKind.var_positional:
        prefix = "*"
    elif param.kind == ParameterKind.var_keyword:
        prefix = "**"
    text = f"{prefix}{param.name}"
    if param.annotation is not None:
        text += f": {param.annotation}"
    if param.default is not None and not prefix:
        text += f" = {param.default}"
    return text


def _iter_public_methods(cls: Class) -> Iterator[Method]:
    """Yield every public method of ``cls`` (plus ``__init__``) as :class:`Method`."""
    for name, member in cls.members.items():
        if name.startswith("_") and name != "__init__":
            continue
        target = _try_resolve(member)
        if not isinstance(target, Function):
            continue
        source_path = _relpath(target.filepath)
        source_line = int(target.lineno or 0)
        yield Method(
            name=name,
            signature=_function_signature(target),
            summary=_first_docstring_line(target),
            source_path=source_path,
            source_line=source_line,
            source_href=_source_href(source_path, source_line),
            docs=_docstring_sections(target),
        )


def _relpath(path: object) -> str:
    """Repo-relative POSIX path for a griffe ``filepath`` value."""
    if path is None:
        return ""
    file_path = Path(str(path)).resolve()
    try:
        return file_path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return file_path.as_posix()


def _source_href(source_path: str, line: int) -> str:
    """Absolute GitHub URL for a repo-relative source path (optional ``#Lline``)."""
    if not source_path:
        return ""
    suffix = f"#L{line}" if line > 0 else ""
    return f"{SOURCE_BASE}/{source_path}{suffix}"
