#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate a single consolidated Python API reference page using griffe.

Produces a Fern-compatible Markdown file at docs/api/python/README.md
with classes and functions organized into logical sections.

Usage (from repository root):
    python3 docs/scripts/generate_python_api.py
"""

from __future__ import annotations

import argparse
import re
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from griffe import GriffeLoader
except ImportError:
    sys.exit("griffe is required: pip install griffe")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fern_helpers import (  # noqa: E402
    AUTOGEN_WARNING,
    REPO_ROOT,
    SPDX_HEADER,
    escape_jsx_attr,
    escape_mdx_prose,
    render_details,
    slugify,
)

OUTPUT_DIR = REPO_ROOT / "docs" / "api" / "python"

SEARCH_PATHS = [
    str(REPO_ROOT / "components" / "src"),
    str(REPO_ROOT / "lib" / "bindings" / "python" / "src"),
]


# ---------------------------------------------------------------------------
# Section definitions (single source of truth)
# ---------------------------------------------------------------------------


@dataclass
class SectionDef:
    """Defines a logical section of the API reference."""

    title: str
    description: str
    icon: str = "regular code"
    classes: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    module: str | None = None  # render all public members from this module


SECTION_DEFS: list[SectionDef] = [
    SectionDef(
        title="Runtime Core",
        description="Core runtime primitives for building distributed Dynamo applications.",
        icon="regular microchip",
        classes=[
            "DistributedRuntime",
            "Component",
            "Endpoint",
            "Client",
            "Context",
            "JsonLike",
            "ModelCardInstanceId",
        ],
    ),
    SectionDef(
        title="HTTP and gRPC Services",
        description="Network service wrappers for exposing models via HTTP (OpenAI-compatible) and gRPC (KServe).",
        icon="regular globe",
        classes=[
            "HttpService",
            "KserveGrpcService",
            "PythonAsyncEngine",
            "HttpAsyncEngine",
        ],
    ),
    SectionDef(
        title="Model Management",
        description="Register, unregister, and fetch models from the distributed runtime.",
        icon="regular cube",
        classes=[
            "ModelDeploymentCard",
            "ModelRuntimeConfig",
            "ModelInput",
            "ModelType",
        ],
        functions=[
            "register_model",
            "unregister_model",
            "fetch_model",
            "register_llm",
            "unregister_llm",
            "fetch_llm",
            "lora_name_to_id",
        ],
    ),
    SectionDef(
        title="KV Cache Routing",
        description="KV-aware routing for prefix-optimized request placement across workers.",
        icon="regular route",
        classes=[
            "KvRouter",
            "KvRouterConfig",
            "RouterMode",
            "RouterConfig",
            "KvIndexer",
            "ApproxKvIndexer",
            "RadixTree",
            "OverlapScores",
        ],
        functions=["compute_block_hash_for_seq"],
    ),
    SectionDef(
        title="KV Cache Memory",
        description="Block-level KV cache memory management.",
        icon="regular memory",
        classes=["BlockManager", "Block", "BlockList", "Layer", "KvbmRequest"],
    ),
    SectionDef(
        title="KV Events and Metrics",
        description="Publish KV cache events and worker load metrics for routing decisions.",
        icon="regular chart-line",
        classes=["KvEventPublisher", "WorkerMetricsPublisher"],
    ),
    SectionDef(
        title="Planner",
        description="Scaling connectors and decision types for the Dynamo Planner.",
        icon="regular arrows-split-up-and-left",
        classes=[
            "PlannerDecision",
            "VirtualConnectorCoordinator",
            "VirtualConnectorClient",
            "PlannerConnector",
            "KubernetesConnector",
            "VirtualConnector",
            "GlobalPlannerConnector",
            "SLAPlannerDefaults",
            "TargetReplica",
            "SubComponentType",
            "ScaleRequestHandler",
        ],
    ),
    SectionDef(
        title="NIXL Connect",
        description="RDMA-based data transfer operations via the NIXL library.",
        icon="regular network-wired",
        module="dynamo.nixl_connect",
    ),
    SectionDef(
        title="Configuration and Utilities",
        description="Engine configuration, entrypoint arguments, and shared enums.",
        icon="regular gear",
        classes=["EngineConfig", "EntrypointArgs", "EngineType", "RuntimeMetrics"],
        functions=["make_engine", "run_input", "log_message"],
    ),
]


# ---------------------------------------------------------------------------
# Docstring parsing (Google-style)
# ---------------------------------------------------------------------------

_PARAM_EMPTY_SENTINEL = "inspect.Parameter.empty"

_SECTION_KW_RE = re.compile(
    r"^(Args|Arguments|Parameters|Returns?|Raises?|Throws"
    r"|Examples?|Notes?|Attributes|Yields|See Also|Warnings?):\s*$",
    re.MULTILINE,
)

_SECTION_KEY_MAP: dict[str, str] = {
    "args": "args",
    "arguments": "args",
    "parameters": "args",
    "return": "returns",
    "returns": "returns",
    "raise": "raises",
    "raises": "raises",
    "throws": "raises",
    "example": "example",
    "examples": "example",
    "note": "note",
    "notes": "note",
    "attributes": "attributes",
    "yields": "yields",
    "see also": "see_also",
    "warning": "warning",
    "warnings": "warning",
}


def _parse_docstring(raw: str) -> dict[str, str]:
    """Parse a Google-style docstring into named sections."""
    if not raw:
        return {}
    raw = raw.strip()
    matches = list(_SECTION_KW_RE.finditer(raw))
    if not matches:
        return {"description": raw}

    sections: dict[str, str] = {}
    desc = raw[: matches[0].start()].strip()
    if desc:
        sections["description"] = desc

    for i, match in enumerate(matches):
        keyword = match.group(1).lower()
        key = _SECTION_KEY_MAP.get(keyword, keyword)
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        body = _extract_section_body(raw[match.end() : end])
        if key in sections and key != "description":
            sections[key] += "\n\n" + body
        else:
            sections[key] = body

    return sections


def _extract_section_body(text: str) -> str:
    """Extract only the indented portion of a section body."""
    lines = text.split("\n")
    body_lines: list[str] = []
    found_indented = False
    for line in lines:
        stripped = line.rstrip()
        if not stripped:
            body_lines.append("")
            continue
        indent = len(line) - len(line.lstrip())
        if indent > 0:
            found_indented = True
            body_lines.append(line)
        elif found_indented:
            break
    return textwrap.dedent("\n".join(body_lines)).strip()


def _parse_indented_items(text: str) -> list[str]:
    """Parse indented items from a Google-style section body."""
    items: list[str] = []
    current: list[str] = []
    base_indent: int | None = None

    for line in text.split("\n"):
        stripped = line.rstrip()
        if not stripped:
            continue
        indent = len(line) - len(line.lstrip())
        if base_indent is None:
            base_indent = indent
        if indent <= base_indent and current:
            items.append(" ".join(current))
            current = [stripped.strip()]
        else:
            current.append(stripped.strip())
    if current:
        items.append(" ".join(current))
    return items


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _escape_table(text: str) -> str:
    """Minimal escaping for Markdown table cells."""
    return text.replace("|", "\\|").replace("\n", " ")


def _raw_first_line(obj: Any) -> str:
    """Get the first line of an object's docstring, unescaped."""
    if obj.docstring and obj.docstring.value:
        return obj.docstring.value.strip().split("\n")[0].strip()
    return ""


def _format_annotation(annotation: Any) -> str:
    """Best-effort annotation to string (no escaping — used in code spans)."""
    if annotation is None:
        return ""
    return str(annotation)


def _resolve(obj: Any) -> Any:
    """Resolve griffe Alias objects to their target."""
    try:
        if hasattr(obj, "resolve_target"):
            obj.resolve_target()
        return obj.target if hasattr(obj, "target") and obj.target is not None else obj
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
# Structured docstring renderers
# ---------------------------------------------------------------------------


def _render_description(
    parsed: dict[str, str], *, skip_first_line: bool = False
) -> str:
    """Render the description section with MDX escaping."""
    desc = parsed.get("description", "")
    if not desc:
        return ""
    if skip_first_line:
        lines = desc.split("\n", 1)
        desc = lines[1].strip() if len(lines) > 1 else ""
    if not desc:
        return ""
    return escape_mdx_prose(desc)


def _render_raises(parsed: dict[str, str]) -> str:
    """Render Raises section as a Markdown bullet list."""
    raw = parsed.get("raises", "")
    if not raw:
        return ""
    parts = ["**Raises**\n"]
    for item in _parse_indented_items(raw):
        name, sep, desc = item.partition(": ")
        if sep and name.strip():
            parts.append(f"- `{name.strip()}` -- {escape_mdx_prose(desc.strip())}")
        else:
            parts.append(f"- {escape_mdx_prose(item.strip())}")
    parts.append("")
    return "\n".join(parts)


def _render_example(parsed: dict[str, str]) -> str:
    """Render Example section inside a Python code fence."""
    raw = parsed.get("example", "")
    if not raw:
        return ""
    dedented = textwrap.dedent(raw)
    return f"**Example**\n\n```python\n{dedented.strip()}\n```\n"


def _render_notes(parsed: dict[str, str]) -> str:
    """Render Note/Warning sections as GitHub callouts."""
    parts: list[str] = []
    note = parsed.get("note", "")
    if note:
        escaped = escape_mdx_prose(textwrap.dedent(note).strip())
        parts.append(f"> [!NOTE]\n> {escaped}\n")
    warning = parsed.get("warning", "")
    if warning:
        escaped = escape_mdx_prose(textwrap.dedent(warning).strip())
        parts.append(f"> [!WARNING]\n> {escaped}\n")
    return "\n".join(parts)


def _render_parameters_table(params: Any) -> str:
    """Render a function's parameters as a Markdown table."""
    rows: list[str] = []
    for param in params:
        if param.name in ("self", "cls"):
            continue
        annotation = _format_annotation(param.annotation)
        default = str(param.default) if param.default is not None else ""
        if default == _PARAM_EMPTY_SENTINEL:
            default = ""
        rows.append(f"| `{param.name}` | `{annotation}` | {_escape_table(default)} |")
    if not rows:
        return ""
    header = "| Parameter | Type | Default |\n| --- | --- | --- |"
    return header + "\n" + "\n".join(rows) + "\n"


def _build_signature(func: Any) -> str:
    """Build the function signature string."""
    sig_params: list[str] = []
    try:
        params = func.parameters
    except Exception as exc:
        print(f"  WARN: no parameters for {func}: {exc}", file=sys.stderr)
        params = []
    for p in params:
        if p.name in ("self", "cls"):
            continue
        ann = _format_annotation(p.annotation)
        entry = f"{p.name}: {ann}" if ann else p.name
        if p.default is not None and str(p.default) != _PARAM_EMPTY_SENTINEL:
            entry += f" = {p.default}"
        sig_params.append(entry)
    ret = _format_annotation(getattr(func, "returns", None))
    sig = f"({', '.join(sig_params)})"
    if ret:
        sig += f" -> {ret}"
    return sig


def _render_returns(parsed: dict[str, str], func: Any) -> str:
    """Render the Returns section, merging annotation with docstring."""
    ret = _format_annotation(getattr(func, "returns", None))
    if not ret:
        return ""
    returns_body = parsed.get("returns", "")
    desc_text = _extract_returns_desc(returns_body)
    if desc_text and desc_text.lower() != ret.lower():
        return f"**Returns:** `{ret}` -- {escape_mdx_prose(desc_text)}\n"
    return f"**Returns:** `{ret}`\n"


def _extract_returns_desc(returns_body: str) -> str:
    """Extract description text from a Returns docstring section."""
    if not returns_body:
        return ""
    first_line = returns_body.strip().split("\n")[0].strip()
    if ": " in first_line:
        return first_line.split(": ", 1)[1].strip()
    return first_line


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

    sig = _build_signature(func)
    is_async = getattr(func, "is_async", False)
    prefix = "async " if is_async else ""
    parts.append(f"{heading_level} `{prefix}{func.name}{sig}`\n")

    raw_doc = ""
    if func.docstring and func.docstring.value:
        raw_doc = func.docstring.value.strip()
    parsed = _parse_docstring(raw_doc)

    desc = _render_description(parsed, skip_first_line=skip_first_line)
    if desc:
        parts.append(desc + "\n")

    try:
        params = func.parameters
    except Exception as exc:
        print(f"  WARN: no parameters for {func}: {exc}", file=sys.stderr)
        params = []
    params_table = _render_parameters_table(params)
    if params_table:
        parts.append("**Parameters**\n")
        parts.append(params_table)

    returns = _render_returns(parsed, func)
    if returns:
        parts.append(returns)

    for renderer in (_render_raises, _render_example, _render_notes):
        section = renderer(parsed)
        if section:
            parts.append(section)

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

    raw_doc = ""
    if cls.docstring and cls.docstring.value:
        raw_doc = cls.docstring.value.strip()
    parsed = _parse_docstring(raw_doc)
    desc = _render_description(parsed, skip_first_line=skip_first_line)
    if desc:
        parts.append(desc + "\n")

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
    parts.append("**Attributes**\n")
    for attr in attrs:
        parts.append(_render_attribute(attr))
    parts.append("")


def _append_constructor(parts: list[str], members: Any) -> None:
    """Append constructor rendering to parts if __init__ or __new__ exists."""
    constructor = members.get("__init__") or members.get("__new__")
    if not constructor:
        return
    parts.append("**Constructor**\n")
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
    parts.append("**Methods**\n")
    for method in methods:
        parts.append(_render_function(method, heading_level="####"))


def _render_class_details(cls: Any) -> str:
    """Render a class wrapped in a <details>/<summary> block."""
    cls = _resolve(cls)
    summary_text = _raw_first_line(cls)
    title = (
        escape_jsx_attr(f"{cls.name} — {summary_text}") if summary_text else cls.name
    )
    body = _render_class_body(cls, skip_first_line=True)
    return render_details(title, body)


def _render_function_details(func: Any) -> str:
    """Render a standalone function wrapped in a <details>/<summary> block."""
    func = _resolve(func)
    summary_text = _raw_first_line(func)
    title = (
        escape_jsx_attr(f"{func.name}() — {summary_text}")
        if summary_text
        else f"{func.name}()"
    )
    body = _render_function(func, heading_level="####", skip_first_line=True)
    return render_details(title, body)


# ---------------------------------------------------------------------------
# Loading and classification
# ---------------------------------------------------------------------------


def _discover_modules() -> list[str]:
    """Find all dynamo.* submodules across search paths."""
    found: set[str] = set()
    for sp in SEARCH_PATHS:
        dynamo_dir = Path(sp) / "dynamo"
        if not dynamo_dir.is_dir():
            continue
        for child in sorted(dynamo_dir.iterdir()):
            if child.is_dir() and (child / "__init__.py").exists():
                if not child.name.startswith("_"):
                    found.add(f"dynamo.{child.name}")
    found.add("dynamo._core")
    return sorted(found)


def _create_loader() -> GriffeLoader:
    """Create a GriffeLoader with the correct search paths."""
    return GriffeLoader(search_paths=SEARCH_PATHS)


def _load_all_members(
    loader: GriffeLoader,
    modules: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load all modules and collect classes and functions by name."""
    all_classes: dict[str, Any] = {}
    all_functions: dict[str, Any] = {}

    for module_name in modules:
        try:
            mod = loader.load(module_name)
        except Exception as exc:
            print(f"  SKIP {module_name}: {exc}", file=sys.stderr)
            continue
        print(f"  Loaded: {module_name}")

        for member in mod.members.values():
            kind = _safe_kind(member)
            if member.name.startswith("_"):
                continue
            if kind == "CLASS":
                all_classes[member.name] = member
            elif kind == "FUNCTION":
                all_functions[member.name] = member

    loader.resolve_aliases()
    return all_classes, all_functions


def _extract_nixl_members(
    loader: GriffeLoader,
) -> tuple[list[Any], list[Any]]:
    """Extract classes and functions from dynamo.nixl_connect."""
    classes: list[Any] = []
    functions: list[Any] = []
    try:
        mod = loader.modules_collection["dynamo.nixl_connect"]
    except KeyError:
        print("  WARN: dynamo.nixl_connect not in collection", file=sys.stderr)
        return classes, functions

    for member in mod.members.values():
        kind = _safe_kind(member)
        if member.name.startswith("_"):
            continue
        if kind == "CLASS":
            classes.append(member)
        elif kind == "FUNCTION":
            functions.append(member)
    return classes, functions


# ---------------------------------------------------------------------------
# Page rendering
# ---------------------------------------------------------------------------


def _render_header() -> list[str]:
    """Render the page header with SPDX, title, and card navigation."""
    parts: list[str] = [
        SPDX_HEADER.format(sidebar_title="Python API"),
        AUTOGEN_WARNING,
        "# Python API Reference\n",
        "This page documents the public Python API for NVIDIA Dynamo. "
        "Classes and functions are organized by functional area. "
        "Expand any item to see its full API.\n",
    ]

    parts.append("<CardGroup cols={3}>\n")
    for section in SECTION_DEFS:
        anchor = slugify(section.title)
        parts.append(
            f'  <Card title="{section.title}" icon="{section.icon}" href="#{anchor}">\n'
            f"    {section.description}\n"
            f"  </Card>\n"
        )
    parts.append("</CardGroup>\n")
    return parts


def _render_section(
    section: SectionDef,
    all_classes: dict[str, Any],
    all_functions: dict[str, Any],
    nixl_classes: list[Any],
    nixl_functions: list[Any],
    placed_classes: set[str],
    placed_functions: set[str],
) -> list[str]:
    """Render a single section of the API reference."""
    parts: list[str] = [
        "---\n",
        f"## {section.title}\n",
        f"{section.description}\n",
    ]

    if section.module:
        _render_module_section(
            parts,
            nixl_classes,
            nixl_functions,
            placed_classes,
            placed_functions,
        )
        return parts

    items = _collect_section_items(
        section,
        all_classes,
        all_functions,
        placed_classes,
        placed_functions,
    )
    parts.extend(items)
    return parts


def _render_module_section(
    parts: list[str],
    nixl_classes: list[Any],
    nixl_functions: list[Any],
    placed_classes: set[str],
    placed_functions: set[str],
) -> None:
    """Render all members from a whole-module section (e.g., NIXL Connect)."""
    for cls in nixl_classes:
        parts.append(_render_class_details(cls))
        placed_classes.add(cls.name)
    for func in nixl_functions:
        parts.append(_render_function_details(func))
        placed_functions.add(func.name)


def _collect_section_items(
    section: SectionDef,
    all_classes: dict[str, Any],
    all_functions: dict[str, Any],
    placed_classes: set[str],
    placed_functions: set[str],
) -> list[str]:
    """Collect rendered items for a class+function section."""
    items: list[str] = []
    for class_name in section.classes:
        if class_name in all_classes:
            items.append(_render_class_details(all_classes[class_name]))
            placed_classes.add(class_name)

    for func_name in section.functions:
        if func_name in all_functions:
            items.append(_render_function_details(all_functions[func_name]))
            placed_functions.add(func_name)

    return items


def _render_remaining(
    all_classes: dict[str, Any],
    all_functions: dict[str, Any],
    placed_classes: set[str],
    placed_functions: set[str],
) -> list[str]:
    """Render any classes/functions not assigned to a section."""
    remaining_classes = {
        n: o for n, o in all_classes.items() if n not in placed_classes
    }
    remaining_functions = {
        n: o for n, o in all_functions.items() if n not in placed_functions
    }
    if not remaining_classes and not remaining_functions:
        return []

    parts: list[str] = ["---\n", "## Other\n"]
    for cls in remaining_classes.values():
        parts.append(_render_class_details(cls))
    for func in remaining_functions.values():
        parts.append(_render_function_details(func))
    return parts


def render_consolidated_page() -> str:
    """Render the single consolidated Python API reference page."""
    modules = _discover_modules()
    print(f"Loading modules ({len(modules)} discovered)...")
    loader = _create_loader()
    all_classes, all_functions = _load_all_members(loader, modules)
    nixl_classes, nixl_functions = _extract_nixl_members(loader)

    placed_classes: set[str] = set()
    placed_functions: set[str] = set()

    parts: list[str] = _render_header()

    for section in SECTION_DEFS:
        parts.extend(
            _render_section(
                section,
                all_classes,
                all_functions,
                nixl_classes,
                nixl_functions,
                placed_classes,
                placed_functions,
            )
        )

    parts.extend(
        _render_remaining(
            all_classes,
            all_functions,
            placed_classes,
            placed_functions,
        )
    )

    parts.append("---\n")
    parts.append("*Source packages: " + ", ".join(f"`{m}`" for m in modules) + "*\n")

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
