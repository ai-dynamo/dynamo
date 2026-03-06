#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate a single consolidated Python API reference page using griffe.

Produces a GitHub-friendly Markdown file at docs/api/python/README.md
with classes and functions organized by module. The output includes
YAML frontmatter for Fern compatibility but uses only standard Markdown
constructs (details/summary, tables).

Usage (from repository root):
    python3 docs/scripts/generate_python_api.py
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from griffe import DocstringSectionKind, GriffeLoader
except ImportError:
    sys.exit("griffe>=1.0 is required: pip install griffe")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fern_helpers import (  # noqa: E402
    REPO_ROOT,
    details_title,
    escape_mdx_prose,
    render_details,
    render_markdown_table,
    slugify,
)

OUTPUT_DIR = REPO_ROOT / "docs" / "api" / "python"

SEARCH_PATHS = [
    str(REPO_ROOT / "components" / "src"),
    str(REPO_ROOT / "lib" / "bindings" / "python" / "src"),
]

# ---------------------------------------------------------------------------
# Module configuration
# ---------------------------------------------------------------------------

MODULE_ORDER: list[str] = [
    "dynamo._core",
    "dynamo.runtime",
    "dynamo.llm",
    "dynamo.frontend",
    "dynamo.common",
    "dynamo.health_check",
    "dynamo.logits_processing",
    "dynamo.planner",
    "dynamo.router",
    "dynamo.mocker",
    "dynamo.nixl_connect",
]

MODULE_WHITELIST: set[str] = set(MODULE_ORDER)

MODULE_ICONS: dict[str, str] = {
    "dynamo._core": "regular microchip",
    "dynamo.runtime": "regular play",
    "dynamo.llm": "regular brain",
    "dynamo.frontend": "regular globe",
    "dynamo.common": "regular toolbox",
    "dynamo.health_check": "regular heart-pulse",
    "dynamo.logits_processing": "regular sliders",
    "dynamo.planner": "regular chart-line",
    "dynamo.router": "regular route",
    "dynamo.mocker": "regular vial",
    "dynamo.nixl_connect": "regular network-wired",
}

MODULE_DESCRIPTIONS: dict[str, str] = {
    "dynamo._core": (
        "Low-level Rust-backed runtime, routing, KV cache, memory, "
        "and model management bindings."
    ),
    "dynamo.runtime": (
        "Decorators and utilities for defining Dynamo workers and endpoints."
    ),
    "dynamo.llm": "LLM serving pipeline components and engine integration.",
    "dynamo.frontend": (
        "HTTP frontend configuration and OpenAI-compatible API gateway."
    ),
    "dynamo.common": (
        "Shared configuration, constants, storage, and utility functions."
    ),
    "dynamo.health_check": (
        "Health check payload and environment-based configuration."
    ),
    "dynamo.logits_processing": ("Custom logits processing for LLM token generation."),
    "dynamo.planner": ("Scaling connectors and decision types for the Dynamo Planner."),
    "dynamo.router": "Request routing configuration and argument groups.",
    "dynamo.mocker": "Mock engine for testing without GPU resources.",
    "dynamo.nixl_connect": (
        "RDMA-based data transfer operations via the NIXL library."
    ),
}


# ---------------------------------------------------------------------------
# Sub-grouping (Enhancement 4)
# ---------------------------------------------------------------------------

SUB_GROUP_ORDER = ["enums", "configuration", "classes", "data_models", "functions"]

SUB_GROUP_TITLES: dict[str, str] = {
    "enums": "Enums",
    "configuration": "Configuration",
    "classes": "Classes",
    "data_models": "Data Models",
    "functions": "Functions",
}


# ---------------------------------------------------------------------------
# Bug fix 1: strip raw NumPy-style parameter blocks from text sections
# ---------------------------------------------------------------------------

_NUMPYDOC_SECTION_RE = re.compile(
    r"(?:Parameters|Returns|Raises|Yields|Attributes|Notes|References|Examples)"
    r"\s*:?\s*\n"
    r"\s*-{3,}\s*\n"
    r"[\s\S]*?(?=\n\n|\Z)",
)


# ---------------------------------------------------------------------------
# Constants and context
# ---------------------------------------------------------------------------

_PARAM_EMPTY_SENTINEL = "inspect.Parameter.empty"

FRONTMATTER = """\
---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Python API
max-toc-depth: 2
---
"""


@dataclass
class RenderContext:
    """Mutable state passed through the rendering pipeline."""

    seen_names: set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _module_display_name(module_name: str) -> str:
    """Convert module path to display name."""
    if module_name == "dynamo._core":
        return "Core Bindings"
    return module_name


def _strip_raw_param_blocks(text: str) -> str:
    """Remove NumPy-style parameter/returns blocks that griffe didn't parse."""
    return _NUMPYDOC_SECTION_RE.sub("", text).strip()


def _escape_table(text: str) -> str:
    """Minimal escaping for Markdown table cells."""
    return text.replace("|", "\\|").replace("\n", " ")


def _raw_first_line(obj: Any) -> str:
    """Get the first line of an object's docstring, unescaped."""
    if obj.docstring and obj.docstring.value:
        return obj.docstring.value.strip().split("\n")[0].strip()
    return ""


def _format_annotation(annotation: Any) -> str:
    """Best-effort annotation to string (no escaping)."""
    if annotation is None:
        return ""
    text = str(annotation)
    if not text or text == "None":
        return ""
    return text


def _resolve(obj: Any) -> Any:
    """Resolve griffe Alias objects to their target."""
    try:
        if hasattr(obj, "resolve_target"):
            obj.resolve_target()
        target = getattr(obj, "target", None)
        return target if target is not None else obj
    except Exception as exc:
        print(f"  WARN: could not resolve {obj}: {exc}", file=sys.stderr)
        return obj


def _safe_kind(member: Any) -> str:
    """Get the kind name of a member, resolving aliases safely."""
    try:
        return member.kind.name
    except Exception as exc:
        print(f"  WARN: unknown kind for {member}: {exc}", file=sys.stderr)
        return "UNKNOWN"


# ---------------------------------------------------------------------------
# Enhancement 1: Source links
# ---------------------------------------------------------------------------


def _source_link(obj: Any) -> str:
    """Generate a source file reference."""
    try:
        filepath = obj.filepath
        if filepath is None:
            return ""
        repo_rel = str(Path(filepath).relative_to(REPO_ROOT))
        display_rel = repo_rel
        for prefix in SEARCH_PATHS:
            try:
                display_rel = str(Path(filepath).relative_to(prefix))
                break
            except ValueError:
                continue
        lineno = getattr(obj, "lineno", None)
        suffix = f"#L{lineno}" if lineno else ""
        return (
            f"*Source: [`{display_rel}{suffix}`]"
            f"(https://github.com/ai-dynamo/dynamo/blob/main/{repo_rel}{suffix})*\n"
        )
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Enhancement 2: Inheritance display
# ---------------------------------------------------------------------------


def _get_bases(cls: Any) -> list[str]:
    """Get base class names, filtering out object and ABC."""
    try:
        bases = cls.bases
        if not bases:
            return []
        return [str(b) for b in bases if str(b) not in ("object", "ABC")]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Enhancement 4: Member classification for sub-grouping
# ---------------------------------------------------------------------------


def _classify_member(member: Any) -> str:
    """Classify a member into a sub-group for display ordering."""
    try:
        if member.kind.name == "FUNCTION":
            return "functions"
        bases = [str(b) for b in (getattr(member, "bases", None) or [])]
        base_names = [b.split(".")[-1] for b in bases]
        if any(n in base_names for n in ("Enum", "IntEnum", "StrEnum")):
            return "enums"
        if any(n in base_names for n in ("ArgGroup", "ConfigBase", "BaseModel")):
            return "configuration"
        labels = getattr(member, "labels", None)
        if labels and "dataclass" in labels:
            return "data_models"
        name = member.name
        suffixes = (
            "Request",
            "Response",
            "Result",
            "Data",
            "Info",
            "Entry",
            "Card",
        )
        if any(name.endswith(s) for s in suffixes):
            return "data_models"
        return "classes"
    except Exception:
        return "classes"


# ---------------------------------------------------------------------------
# Docstring rendering (griffe-native)
# ---------------------------------------------------------------------------


def _render_docstring_sections(
    docstring: Any,
    *,
    skip_first_line: bool = False,
    fallback_params: Any = None,
) -> str:
    """Render griffe-parsed docstring sections to Markdown."""
    if not docstring or not docstring.value:
        return ""

    try:
        sections = docstring.parsed
    except Exception:
        text = docstring.value.strip()
        if skip_first_line:
            lines = text.split("\n", 1)
            text = lines[1].strip() if len(lines) > 1 else ""
        return escape_mdx_prose(text) if text else ""

    # Pre-scan: check for params section, collect all returns (Bug 2)
    has_params = any(s.kind == DocstringSectionKind.parameters for s in sections)
    all_returns: list[tuple[str, str]] = []
    for section in sections:
        if section.kind == DocstringSectionKind.returns and section.value:
            for r in section.value:
                ann = str(r.annotation) if r.annotation else ""
                desc = escape_mdx_prose(r.description or "")
                all_returns.append((ann, desc))

    parts: list[str] = []
    returns_rendered = False
    fallback_inserted = False
    in_description = True

    for section in sections:
        # Inject fallback param table when leaving description zone
        if (
            section.kind != DocstringSectionKind.text
            and in_description
            and not has_params
            and not fallback_inserted
            and fallback_params is not None
        ):
            table = _render_parameters_table(fallback_params)
            if table:
                parts.append(table)
            fallback_inserted = True

        if section.kind != DocstringSectionKind.text:
            in_description = False

        match section.kind:
            case DocstringSectionKind.text:
                text = section.value
                if skip_first_line and not parts:
                    lines = text.split("\n", 1)
                    text = lines[1].strip() if len(lines) > 1 else ""
                text = _strip_raw_param_blocks(text)
                if text:
                    parts.append(escape_mdx_prose(text))

            case DocstringSectionKind.parameters:
                params = section.value
                if params:
                    rows: list[str] = []
                    for p in params:
                        ann = _format_annotation(p.annotation) or "Any"
                        desc = escape_mdx_prose(p.description or "")
                        rows.append(
                            f"| `{p.name}` | `{ann}` " f"| {_escape_table(desc)} |"
                        )
                    if rows:
                        header = (
                            "| Parameter | Type | Description |\n" "| --- | --- | --- |"
                        )
                        parts.append(
                            f"<b>Parameters</b>\n\n{header}\n" + "\n".join(rows)
                        )

            case DocstringSectionKind.returns:
                if returns_rendered:
                    continue
                returns_rendered = True
                if all_returns:
                    ret_lines: list[str] = []
                    for ann, desc in all_returns:
                        if desc:
                            ret_lines.append(f"`{ann}` -- {desc}" if ann else desc)
                        elif ann:
                            ret_lines.append(f"`{ann}`")
                    if len(ret_lines) == 1:
                        parts.append(f"<b>Returns:</b> {ret_lines[0]}")
                    elif ret_lines:
                        parts.append(
                            "<b>Returns:</b>\n\n"
                            + "\n".join(f"- {r}" for r in ret_lines)
                        )

            case DocstringSectionKind.raises:
                raises = section.value
                if raises:
                    items: list[str] = []
                    for r in raises:
                        ann = str(r.annotation) if r.annotation else ""
                        desc = escape_mdx_prose(r.description or "")
                        if ann and desc:
                            items.append(f"- `{ann}` -- {desc}")
                        elif ann:
                            items.append(f"- `{ann}`")
                        elif desc:
                            items.append(f"- {desc}")
                    if items:
                        parts.append("<b>Raises</b>\n\n" + "\n".join(items))

            case DocstringSectionKind.examples:
                examples = section.value
                if examples:
                    example_parts: list[str] = []
                    for kind, content in examples:
                        if kind == DocstringSectionKind.examples:
                            example_parts.append(f"```python\n{content.strip()}\n```")
                        elif kind == DocstringSectionKind.text:
                            stripped = content.strip()
                            if stripped:
                                example_parts.append(escape_mdx_prose(stripped))
                    if example_parts:
                        parts.append("<b>Examples</b>\n\n" + "\n\n".join(example_parts))

            case DocstringSectionKind.admonition:
                adm = section.value
                adm_kind = str(adm.annotation).lower() if adm.annotation else "note"
                raw_desc = adm.description or ""
                if adm_kind == "example":
                    if ">>>" in raw_desc:
                        parts.append(
                            "<b>Example</b>\n\n" f"```python\n{raw_desc.strip()}\n```"
                        )
                    else:
                        parts.append(
                            "<b>Example</b>\n\n" f"{escape_mdx_prose(raw_desc)}"
                        )
                elif adm_kind == "deprecated":
                    desc = escape_mdx_prose(raw_desc)
                    parts.append(f"> [!WARNING]\n> **Deprecated:** {desc}")
                elif adm_kind == "warning":
                    desc = escape_mdx_prose(raw_desc)
                    parts.append(f"> [!WARNING]\n> {desc}")
                else:
                    desc = escape_mdx_prose(raw_desc)
                    parts.append(f"> [!NOTE]\n> {desc}")

            case DocstringSectionKind.attributes:
                attrs = section.value
                if attrs:
                    attr_items: list[str] = []
                    for a in attrs:
                        ann = _format_annotation(a.annotation)
                        desc = escape_mdx_prose(a.description or "")
                        line = f"- `{a.name}`"
                        if ann:
                            line += f": `{ann}`"
                        if desc:
                            line += f" -- {desc}"
                        attr_items.append(line)
                    if attr_items:
                        parts.append("<b>Attributes</b>\n\n" + "\n".join(attr_items))

            case _:
                pass

    # Insert fallback params at end if never inserted
    if not has_params and not fallback_inserted and fallback_params is not None:
        table = _render_parameters_table(fallback_params)
        if table:
            parts.append(table)

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Signature and parameter helpers
# ---------------------------------------------------------------------------


def _safe_parameters(func: Any) -> list:
    """Safely get a function's parameters, returning [] on failure."""
    try:
        return func.parameters
    except Exception as exc:
        print(f"  WARN: no parameters for {func}: {exc}", file=sys.stderr)
        return []


def _build_signature(func: Any, params: list | None = None) -> str:
    """Build the function signature string."""
    if params is None:
        params = _safe_parameters(func)
    sig_params: list[str] = []
    for p in params:
        if p.name in ("self", "cls"):
            continue
        ann = _format_annotation(p.annotation)
        entry = f"{p.name}: {ann}" if ann else p.name
        default = str(p.default)
        if default != _PARAM_EMPTY_SENTINEL:
            entry += f" = {p.default}"
        sig_params.append(entry)
    ret = _format_annotation(getattr(func, "returns", None))
    sig = f"({', '.join(sig_params)})"
    if ret:
        sig += f" -> {ret}"
    return sig


def _render_parameters_table(params: Any) -> str:
    """Render a function's parameters as a 4-column table (from signature)."""
    rows: list[str] = []
    for param in params:
        if param.name in ("self", "cls"):
            continue
        annotation = _format_annotation(param.annotation) or "Any"
        default = str(param.default)
        if default == _PARAM_EMPTY_SENTINEL:
            default = ""
        rows.append(
            f"| `{param.name}` | `{annotation}` " f"| {_escape_table(default)} |  |"
        )
    if not rows:
        return ""
    header = (
        "| Parameter | Type | Default | Description |\n" "| --- | --- | --- | --- |"
    )
    return f"<b>Parameters</b>\n\n{header}\n" + "\n".join(rows) + "\n"


# ---------------------------------------------------------------------------
# Markdown rendering: functions, attributes, classes
# ---------------------------------------------------------------------------


def _render_function(
    func: Any,
    heading_level: str = "####",
    *,
    skip_first_line: bool = False,
) -> str:
    """Render a single function/method."""
    func = _resolve(func)
    parts: list[str] = []
    params = _safe_parameters(func)

    sig = _build_signature(func, params)
    is_async = getattr(func, "is_async", False)
    prefix = "async " if is_async else ""
    parts.append(f"{heading_level} `{prefix}{func.name}{sig}`\n")

    src = _source_link(func)
    if src:
        parts.append(src)

    doc_content = _render_docstring_sections(
        func.docstring,
        skip_first_line=skip_first_line,
        fallback_params=params,
    )
    if doc_content:
        parts.append(doc_content)

    return "\n".join(parts)


def _render_attribute(attr: Any) -> str:
    """Render a module or class-level attribute."""
    ann = _format_annotation(attr.annotation)
    line = f"- `{attr.name}`"
    if ann:
        line += f": `{ann}`"
    doc = _raw_first_line(attr)
    if doc:
        line += f" -- {escape_mdx_prose(doc)}"
    return line


def _render_class_body(cls: Any, *, skip_first_line: bool = False) -> str:
    """Render the inner body of a class (without wrapping heading)."""
    cls = _resolve(cls)
    parts: list[str] = []

    src = _source_link(cls)
    if src:
        parts.append(src)

    doc_content = _render_docstring_sections(
        cls.docstring, skip_first_line=skip_first_line
    )
    if doc_content:
        parts.append(doc_content)

    bases = _get_bases(cls)
    if bases:
        parts.append(f"<b>Bases:</b> {', '.join(f'`{b}`' for b in bases)}\n")

    try:
        members = cls.members
    except Exception as exc:
        print(f"  WARN: no members for {cls}: {exc}", file=sys.stderr)
        return "\n".join(parts)

    _append_attributes(parts, members)
    _append_constructor(parts, members)
    _append_methods(parts, members)

    return "\n".join(parts)


def _append_attributes(parts: list[str], members: Any) -> None:
    """Append attribute list to parts if any exist."""
    attrs = [
        m
        for m in members.values()
        if _safe_kind(m) == "ATTRIBUTE" and not m.name.startswith("_")
    ]
    if not attrs:
        return
    parts.append("<b>Attributes</b>\n")
    for attr in attrs:
        parts.append(_render_attribute(attr))
    parts.append("")


def _append_constructor(parts: list[str], members: Any) -> None:
    """Append constructor rendering to parts if __init__ or __new__ exists."""
    constructor = members.get("__init__") or members.get("__new__")
    if not constructor:
        return
    parts.append("<b>Constructor</b>\n")
    parts.append(_render_function(constructor, heading_level="####"))


def _append_methods(parts: list[str], members: Any) -> None:
    """Append public method renderings to parts."""
    methods = [
        m
        for m in members.values()
        if _safe_kind(m) == "FUNCTION" and not m.name.startswith("_")
    ]
    if not methods:
        return
    parts.append("<b>Methods</b>\n")
    for method in methods:
        parts.append(_render_function(method, heading_level="####"))


def _render_class_details(cls: Any) -> str:
    """Render a class wrapped in an expandable <details> block."""
    cls = _resolve(cls)
    first_line = _raw_first_line(cls)
    title = details_title(cls.name, first_line)
    body = _render_class_body(cls, skip_first_line=True)
    return render_details(title, body)


def _render_function_details(func: Any) -> str:
    """Render a standalone function wrapped in an expandable <details> block."""
    func = _resolve(func)
    first_line = _raw_first_line(func)
    title = details_title(f"{func.name}()", first_line)
    body = _render_function(func, heading_level="####", skip_first_line=True)
    return render_details(title, body)


# ---------------------------------------------------------------------------
# Loading and discovery
# ---------------------------------------------------------------------------


def _discover_modules() -> list[str]:
    """Find all whitelisted dynamo.* submodules across search paths."""
    found: set[str] = set()
    for sp in SEARCH_PATHS:
        dynamo_dir = Path(sp) / "dynamo"
        if not dynamo_dir.is_dir():
            continue
        # Directory-based packages
        for child in sorted(dynamo_dir.iterdir()):
            if child.is_dir() and (child / "__init__.py").exists():
                mod_name = f"dynamo.{child.name}"
                if mod_name in MODULE_WHITELIST:
                    found.add(mod_name)
        # Single-file modules (.py)
        for child in sorted(dynamo_dir.glob("*.py")):
            if child.stem.startswith("__"):
                continue
            mod_name = f"dynamo.{child.stem}"
            if mod_name in MODULE_WHITELIST:
                found.add(mod_name)
        # Type stubs (.pyi) for compiled extensions
        for child in sorted(dynamo_dir.glob("*.pyi")):
            if child.stem.startswith("__"):
                continue
            mod_name = f"dynamo.{child.stem}"
            if mod_name in MODULE_WHITELIST:
                found.add(mod_name)
    return sorted(found)


def _create_loader() -> GriffeLoader:
    """Create a GriffeLoader with the correct search paths."""
    return GriffeLoader(search_paths=SEARCH_PATHS, docstring_parser="auto")


def _discover_module_members(
    module_name: str,
    loader: GriffeLoader,
) -> list[tuple[str, Any]]:
    """Discover public classes and functions in a loaded module."""
    try:
        mod = loader.modules_collection[module_name]
    except KeyError:
        return []
    members: list[tuple[str, Any]] = []
    _collect_members(mod, module_name, members)
    return members


def _collect_members(
    mod: Any,
    prefix: str,
    members: list[tuple[str, Any]],
    *,
    max_depth: int = 1,
    depth: int = 0,
) -> None:
    """Recursively collect public, documented classes and functions."""
    for member in mod.members.values():
        if member.name.startswith("_"):
            continue
        kind = _safe_kind(member)
        if kind in ("CLASS", "FUNCTION"):
            if member.docstring and member.docstring.value:
                members.append((f"{prefix}.{member.name}", member))
        elif kind == "MODULE" and depth < max_depth:
            if member.name == "tests":
                continue
            _collect_members(
                member,
                f"{prefix}.{member.name}",
                members,
                max_depth=max_depth,
                depth=depth + 1,
            )


# ---------------------------------------------------------------------------
# Module section rendering
# ---------------------------------------------------------------------------


def _render_module_section(
    module_name: str,
    members: list[tuple[str, Any]],
    ctx: RenderContext,
) -> list[str]:
    """Render a single module section with optional sub-grouping."""
    desc = MODULE_DESCRIPTIONS.get(module_name, "")
    title = _module_display_name(module_name)
    parts: list[str] = ["---\n", f"## {title}\n", f"{desc}\n"]

    # Group members by kind (Enhancement 4)
    groups: dict[str, list[tuple[str, Any]]] = {k: [] for k in SUB_GROUP_ORDER}
    for full_path, member in members:
        name = full_path.split(".")[-1]
        if name in ctx.seen_names:
            continue
        ctx.seen_names.add(name)
        category = _classify_member(member)
        groups[category].append((full_path, member))

    num_groups = sum(1 for g in SUB_GROUP_ORDER if groups[g])

    for group_key in SUB_GROUP_ORDER:
        group_members = groups[group_key]
        if not group_members:
            continue
        if num_groups > 1:
            parts.append(f"### {SUB_GROUP_TITLES[group_key]}\n")
        for full_path, member in group_members:
            name = full_path.split(".")[-1]
            ctx.seen_names.add(name)
            kind = _safe_kind(member)
            if kind == "CLASS":
                parts.append(_render_class_details(member))
            elif kind == "FUNCTION":
                parts.append(_render_function_details(member))

    return parts


# ---------------------------------------------------------------------------
# Page rendering
# ---------------------------------------------------------------------------


def _render_header() -> list[str]:
    """Render the page header with frontmatter, title, and navigation table."""
    cards = [
        {
            "title": _module_display_name(m),
            "icon": MODULE_ICONS.get(m, "regular code"),
            "href": f"#{slugify(_module_display_name(m))}",
            "description": MODULE_DESCRIPTIONS.get(m, ""),
        }
        for m in MODULE_ORDER
    ]
    intro = (
        "This page documents the public Python API for NVIDIA Dynamo. "
        "Classes and functions are organized by module. "
        "Expand any item to see its full API.\n"
    )
    return [
        FRONTMATTER,
        "# Python API Reference\n",
        intro,
        render_markdown_table(cards),
    ]


def render_consolidated_page() -> str:
    """Render the single consolidated Python API reference page."""
    modules = _discover_modules()
    print(f"Loading modules ({len(modules)} discovered)...")

    loader = _create_loader()
    for module_name in modules:
        try:
            loader.load(module_name)
        except Exception as exc:
            print(f"  SKIP {module_name}: {exc}", file=sys.stderr)
            continue
        print(f"  Loaded: {module_name}")

    loader.resolve_aliases()

    ordered = [m for m in MODULE_ORDER if m in modules]
    ordered.extend(m for m in modules if m not in ordered)

    ctx = RenderContext()
    parts: list[str] = _render_header()

    total_items = 0
    for module_name in ordered:
        members = _discover_module_members(module_name, loader)
        if members:
            total_items += len(members)
            parts.extend(_render_module_section(module_name, members, ctx))

    parts.append("---\n")
    parts.append("*Source packages: " + ", ".join(f"`{m}`" for m in ordered) + "*\n")
    print(f"  Total items: {total_items}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point: parse args and generate the API reference page."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory to write generated Markdown (default: docs/api/python)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    content = render_consolidated_page()
    out_path = args.output_dir / "README.md"
    out_path.write_text(content)
    print(f"  -> {out_path}")

    line_count = content.count("\n")
    print(f"\nGenerated consolidated page ({line_count} lines)")


if __name__ == "__main__":
    main()
