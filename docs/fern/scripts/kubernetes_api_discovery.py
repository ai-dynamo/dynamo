# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic parser for the Dynamo Kubernetes API reference.

The upstream source is
``docs/fern/pages/reference/kubernetes-api/additional-resources/api-reference-k8s.md``
-- crd-ref-docs output stitched with Dynamo's header/footer by
``deploy/operator/Makefile::generate-api-docs`` and post-processed by
``deploy/operator/docs/fix-api-anchors.py`` so v1alpha1 / v1beta1 same-named
types get distinct anchors. That Markdown file is the source of truth for
the CRD API surface; this module turns it into a typed model the docs
generator can render as a compact index without regressing anchors or
losing content.

Design notes:

* State-machine parser over the raw Markdown lines. Regexes are pinned
  to the crd-ref-docs output shape so a schema change trips this parser
  loudly rather than silently drifting.
* Data classes are frozen ``@dataclass``es so downstream renderers can
  rely on the model being immutable across the whole generation pass.
* The parser preserves existing anchors, then prefixes every later
  cross-package collision so all rendered DOM IDs remain unique.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from typing import Literal

from markdown_table_parser import split_markdown_table_row as _split_table_row

TypeKind = Literal["resource", "type", "enum"]

TYPE_H4 = re.compile(r"^####\s+(?P<name>.+?)\s*$", re.MULTILINE)
PACKAGE_H2 = re.compile(r"^##\s+(?P<name>[^\s#].*?)\s*$", re.MULTILINE)
INTRO_H2_SKIP = {"Packages"}
OPERATOR_DEFAULTS_H1 = "# Operator Default Values Injection"
API_REFERENCE_H1 = "# API Reference"
FIELD_ROW_RE = re.compile(
    r"^\|\s*`(?P<name>[^`]+)`\s*(?:_(?P<type>[^_]+(?:_[^_]+)*)_)?\s*\|"
    r"\s*(?P<description>.*?)\s*(?:\|\s*(?P<default>.*?)\s*\|"
    r"\s*(?P<validation>.*?)\s*)?\|\s*$"
)
LINK_RE = re.compile(r"\[(?P<label>[^\]]+)\]\(#(?P<anchor>[^)]+)\)")


@dataclass(frozen=True)
class KubernetesTypeRef:
    """One local-anchor reference to another type in the reference."""

    name: str
    anchor: str


@dataclass(frozen=True)
class KubernetesField:
    """One schema field row (``| \\`name\\` _type_ | description | ... |``)."""

    name: str
    type: str
    default: str
    required: bool
    description: str
    validation: str


@dataclass(frozen=True)
class KubernetesEnumValue:
    """One enum value row (``| \\`Value\\` | description |``)."""

    name: str
    description: str


@dataclass(frozen=True)
class KubernetesType:
    """One documented type (schema, enum, or resource) under a package."""

    name: str
    display_name: str
    anchor: str
    kind: TypeKind
    description: str
    underlying_type: str
    validation: str
    appears_in: tuple[KubernetesTypeRef, ...] = field(default_factory=tuple)
    fields: tuple[KubernetesField, ...] = field(default_factory=tuple)
    enum_values: tuple[KubernetesEnumValue, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class KubernetesPackage:
    """One CRD/config API package (e.g. ``nvidia.com/v1alpha1``)."""

    name: str
    anchor: str
    description: str
    resource_types: tuple[KubernetesTypeRef, ...]
    types: tuple[KubernetesType, ...]


@dataclass(frozen=True)
class OperatorDefaultsSubsection:
    """One ``##``-level subsection of the operator-defaults trailer."""

    title: str
    anchor: str
    body_markdown: str


@dataclass(frozen=True)
class OperatorDefaults:
    """The operator-defaults trailer with its intro + twelve subsections."""

    intro_markdown: str
    subsections: tuple[OperatorDefaultsSubsection, ...]


@dataclass(frozen=True)
class KubernetesReference:
    """Full parsed model of the Kubernetes API reference document."""

    packages: tuple[KubernetesPackage, ...]
    operator_defaults: OperatorDefaults


# ---------------------------------------------------------------------------
# Top-level split
# ---------------------------------------------------------------------------


def parse_reference(source: str) -> KubernetesReference:
    """Parse the full ``api-reference.md`` document into a typed model."""
    package_text, defaults_text = _split_defaults_section(source)
    packages = _unique_type_anchors(tuple(_iter_packages(package_text)))
    operator_defaults = _parse_operator_defaults(defaults_text)
    return KubernetesReference(packages=packages, operator_defaults=operator_defaults)


def _unique_type_anchors(
    packages: tuple[KubernetesPackage, ...],
) -> tuple[KubernetesPackage, ...]:
    """Preserve first anchors and prefix every later cross-package collision."""
    seen: set[str] = set()
    return tuple(_remap_package_anchors(package, seen) for package in packages)


def _remap_package_anchors(
    package: KubernetesPackage, seen: set[str]
) -> KubernetesPackage:
    """Assign unique anchors and rewrite every package-local reference."""
    remap: dict[str, str] = {}
    for type_ in package.types:
        if type_.anchor in remap:
            raise ValueError(
                f"duplicate type anchor within {package.name}: {type_.anchor}"
            )
        candidate = type_.anchor
        if candidate in seen:
            candidate = f"{_package_anchor_prefix(package.name)}-{candidate}"
        if candidate in seen:
            raise ValueError(f"unable to deduplicate type anchor: {candidate}")
        remap[type_.anchor] = candidate
        seen.add(candidate)
    refs = tuple(_remap_type_ref(ref, remap) for ref in package.resource_types)
    types = tuple(_remap_type(type_, remap) for type_ in package.types)
    return replace(package, resource_types=refs, types=types)


def _package_anchor_prefix(package_name: str) -> str:
    """Short stable prefix for a package whose type anchor collided."""
    if package_name.startswith("operator.config."):
        return "operator-config"
    return package_name.rsplit("/", 1)[-1].lower()


def _remap_type(type_: KubernetesType, remap: dict[str, str]) -> KubernetesType:
    """Rewrite one type plus all package-local anchor references it contains."""
    appears_in = tuple(_remap_type_ref(ref, remap) for ref in type_.appears_in)
    fields = tuple(
        replace(field, type=_remap_anchor_links(field.type, remap))
        for field in type_.fields
    )
    return replace(
        type_,
        anchor=remap[type_.anchor],
        appears_in=appears_in,
        fields=fields,
    )


def _remap_type_ref(ref: KubernetesTypeRef, remap: dict[str, str]) -> KubernetesTypeRef:
    """Rewrite a typed local link when its target anchor was prefixed."""
    return replace(ref, anchor=remap.get(ref.anchor, ref.anchor))


def _remap_anchor_links(text: str, remap: dict[str, str]) -> str:
    """Rewrite Markdown links to package-local anchors."""
    return LINK_RE.sub(
        lambda match: (
            f"[{match.group('label')}](#{remap.get(match.group('anchor'), match.group('anchor'))})"
        ),
        text,
    )


def _split_defaults_section(source: str) -> tuple[str, str]:
    """Return ``(packages_body, operator_defaults_body)`` from the source."""
    idx = source.find(OPERATOR_DEFAULTS_H1)
    if idx == -1:
        raise ValueError(
            "api-reference.md missing '# Operator Default Values Injection' trailer"
        )
    return source[:idx], source[idx:]


# ---------------------------------------------------------------------------
# Package parsing
# ---------------------------------------------------------------------------


def _iter_packages(text: str) -> Iterator[KubernetesPackage]:
    """Yield each ``## <package>`` block, skipping the Packages jump list."""
    segments = _split_by_h2(text)
    for name, body in segments:
        if name in INTRO_H2_SKIP:
            continue
        yield _parse_package(name, body)


def _split_by_h2(text: str) -> list[tuple[str, str]]:
    """Split ``text`` into ``[(h2_name, body_up_to_next_h2), ...]`` in order."""
    lines = text.splitlines()
    segments: list[tuple[str, str]] = []
    current_name: str | None = None
    current_lines: list[str] = []
    for line in lines:
        match = PACKAGE_H2.match(line)
        if match:
            if current_name is not None:
                segments.append((current_name, "\n".join(current_lines)))
            current_name = match.group("name").strip()
            current_lines = []
            continue
        if current_name is not None:
            current_lines.append(line)
    if current_name is not None:
        segments.append((current_name, "\n".join(current_lines)))
    return segments


def _parse_package(name: str, body: str) -> KubernetesPackage:
    """Parse one package body into its description + resources + types."""
    intro, types_body = _split_intro_and_types(body)
    description, resource_types = _parse_package_intro(intro)
    types = tuple(_iter_types(types_body))
    return KubernetesPackage(
        name=name,
        anchor=_slugify(name),
        description=description.strip(),
        resource_types=resource_types,
        types=types,
    )


def _split_intro_and_types(body: str) -> tuple[str, str]:
    """Return the pre-``####`` intro and the remaining ``####`` type stream."""
    match = TYPE_H4.search(body)
    if match is None:
        return body, ""
    idx = match.start()
    return body[:idx], body[idx:]


def _parse_package_intro(intro: str) -> tuple[str, tuple[KubernetesTypeRef, ...]]:
    """Return ``(prose, resource_types)`` from a package intro block."""
    prose_lines: list[str] = []
    resource_lines: list[str] = []
    state: Literal["prose", "resources"] = "prose"
    for line in intro.splitlines():
        if state == "prose" and line.strip() == "### Resource Types":
            state = "resources"
            continue
        if state == "resources":
            resource_lines.append(line)
            continue
        prose_lines.append(line)
    resources = _parse_link_list(resource_lines)
    return "\n".join(prose_lines).strip(), resources


def _parse_link_list(lines: list[str]) -> tuple[KubernetesTypeRef, ...]:
    """Return every ``- [Name](#anchor)`` bullet from a list of lines."""
    refs: list[KubernetesTypeRef] = []
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        match = LINK_RE.search(stripped)
        if match is None:
            continue
        refs.append(
            KubernetesTypeRef(name=match.group("label"), anchor=match.group("anchor"))
        )
    return tuple(refs)


# ---------------------------------------------------------------------------
# Type parsing (####)
# ---------------------------------------------------------------------------


def _iter_types(text: str) -> Iterator[KubernetesType]:
    """Split a package body on ``####`` and parse each type block in order."""
    positions = [m.start() for m in TYPE_H4.finditer(text)]
    for idx, start in enumerate(positions):
        end = positions[idx + 1] if idx + 1 < len(positions) else len(text)
        yield _parse_type_block(text[start:end])


def _parse_type_block(block: str) -> KubernetesType:
    """Parse one ``#### TypeName`` block into a :class:`KubernetesType`."""
    header_line, remainder = block.split("\n", 1)
    display_name = TYPE_H4.match(header_line).group("name").strip()
    canonical_name = _strip_v1beta1_prefix(display_name)
    anchor = _type_anchor(display_name, canonical_name)
    parsed = _parse_type_sections(remainder)
    kind = _classify_type(parsed)
    return KubernetesType(
        name=canonical_name,
        display_name=display_name,
        anchor=anchor,
        kind=kind,
        description=parsed.description,
        underlying_type=parsed.underlying_type,
        validation=parsed.validation,
        appears_in=parsed.appears_in,
        fields=parsed.fields if kind != "enum" else (),
        enum_values=parsed.enum_values if kind == "enum" else (),
    )


def _strip_v1beta1_prefix(display: str) -> str:
    """Return the underlying type name for a ``v1beta1 <Name>`` display."""
    return display[len("v1beta1 ") :] if display.startswith("v1beta1 ") else display


def _type_anchor(display: str, canonical: str) -> str:
    """crd-ref-docs anchor slug (with v1beta1 dedup) for a type name."""
    if display.startswith("v1beta1 "):
        return f"v1beta1-{canonical.lower()}"
    return canonical.lower()


@dataclass
class _ParsedTypeSections:
    description: str = ""
    underlying_type: str = ""
    validation: str = ""
    appears_in: tuple[KubernetesTypeRef, ...] = ()
    fields: tuple[KubernetesField, ...] = ()
    enum_values: tuple[KubernetesEnumValue, ...] = ()
    has_default_column: bool = False


def _parse_type_sections(body: str) -> _ParsedTypeSections:
    """Walk one type body and populate the sections we care about."""
    parsed = _ParsedTypeSections()
    lines = body.splitlines()
    prose_buf: list[str] = []
    i = 0
    while i < len(lines):
        i = _consume_next_section(lines, i, parsed, prose_buf)
    parsed.description = _join_prose(prose_buf)
    return parsed


def _consume_next_section(
    lines: list[str], i: int, parsed: _ParsedTypeSections, prose_buf: list[str]
) -> int:
    """Advance ``i`` past one section (or prose line) and mutate ``parsed``."""
    line = lines[i]
    if line.strip() == "_Appears in:_":
        return _consume_appears_in(lines, i + 1, parsed)
    if line.strip() == "_Validation:_":
        return _consume_validation(lines, i + 1, parsed)
    under = _match_underlying_type(line)
    if under is not None:
        parsed.underlying_type = under
        return i + 1
    if line.startswith("| Field | Description"):
        return _consume_table(lines, i, parsed)
    prose_buf.append(line)
    return i + 1


def _match_underlying_type(line: str) -> str | None:
    """Return the underlying-type name (``string``, ``integer``, ...) or None."""
    stripped = line.strip()
    match = re.match(r"^_Underlying type:_\s+_(?P<t>[^_]+)_$", stripped)
    return match.group("t") if match else None


def _consume_appears_in(
    lines: list[str], start: int, parsed: _ParsedTypeSections
) -> int:
    """Consume the ``- [Name](#anchor)`` bullets under ``_Appears in:_``."""
    bullets: list[str] = []
    i = start
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith("- "):
            bullets.append(lines[i])
            i += 1
            continue
        if stripped == "":
            i += 1
            break
        break
    parsed.appears_in = _parse_link_list(bullets)
    return i


def _consume_validation(
    lines: list[str], start: int, parsed: _ParsedTypeSections
) -> int:
    """Consume validation bullets (only ``Enum: [...]`` is captured)."""
    bullets: list[str] = []
    i = start
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith("- "):
            bullets.append(stripped[2:])
            i += 1
            continue
        if stripped == "":
            i += 1
            break
        break
    parsed.validation = "; ".join(bullets)
    return i


def _consume_table(lines: list[str], start: int, parsed: _ParsedTypeSections) -> int:
    """Consume a Markdown table (schema fields or enum values)."""
    header = lines[start]
    parsed.has_default_column = "Default" in header
    i = start + 2  # skip header + separator
    rows: list[str] = []
    while i < len(lines):
        line = lines[i]
        if not line.startswith("|"):
            break
        rows.append(line)
        i += 1
    if parsed.has_default_column:
        parsed.fields = tuple(_parse_field_row(r) for r in rows)
    else:
        parsed.enum_values = tuple(_parse_enum_row(r) for r in rows)
    return i


def _parse_field_row(row: str) -> KubernetesField:
    """Parse one ``| \\`name\\` _type_ | description | default | validation |`` row."""
    cells = _split_table_row(row, 4)
    name = _strip_backticks(cells[0])
    type_ = _extract_type_token(cells[0])
    description = cells[1].strip()
    default = cells[2].strip()
    validation = cells[3].strip()
    required = "Required:" in validation
    return KubernetesField(
        name=name,
        type=type_,
        default=default,
        required=required,
        description=description,
        validation=validation,
    )


def _parse_enum_row(row: str) -> KubernetesEnumValue:
    """Parse one ``| \\`Value\\` | description |`` row (2-column table)."""
    cells = _split_table_row(row, 2)
    return KubernetesEnumValue(
        name=_strip_backticks(cells[0]),
        description=cells[1].strip(),
    )


def _strip_backticks(cell: str) -> str:
    """Return the first backticked identifier in a table cell."""
    match = re.search(r"`([^`]+)`", cell)
    return match.group(1) if match else cell.strip()


def _extract_type_token(cell: str) -> str:
    """Extract the italic type token (``_..._``) that follows the field name."""
    match = re.search(r"`[^`]+`\s+_(.+?)_", cell)
    return match.group(1).strip() if match else ""


def _join_prose(lines: list[str]) -> str:
    """Collapse the free-form paragraph lines into a trimmed prose block."""
    return "\n".join(lines).strip()


def _classify_type(parsed: _ParsedTypeSections) -> TypeKind:
    """Classify a type block as a resource, an enum, or a plain schema."""
    if parsed.enum_values:
        return "enum"
    for f in parsed.fields:
        if f.name == "apiVersion" and f.description.startswith("`"):
            return "resource"
    return "type"


# ---------------------------------------------------------------------------
# Operator defaults trailer
# ---------------------------------------------------------------------------


def _parse_operator_defaults(body: str) -> OperatorDefaults:
    """Parse the ``# Operator Default Values Injection`` trailer."""
    if not body.startswith(OPERATOR_DEFAULTS_H1):
        raise ValueError("operator-defaults section missing expected H1 header")
    tail = body[len(OPERATOR_DEFAULTS_H1) :].lstrip("\n")
    intro, subsections_text = _split_intro_and_subsections(tail)
    subsections = tuple(_iter_operator_subsections(subsections_text))
    return OperatorDefaults(intro_markdown=intro, subsections=subsections)


def _split_intro_and_subsections(text: str) -> tuple[str, str]:
    """Return ``(intro_markdown, rest_starting_at_first_h2)`` from the trailer."""
    match = PACKAGE_H2.search(text)
    if match is None:
        return text.strip(), ""
    idx = match.start()
    return text[:idx].strip(), text[idx:]


def _iter_operator_subsections(text: str) -> Iterator[OperatorDefaultsSubsection]:
    """Yield each ``## <title>`` operator-defaults subsection in order."""
    segments = _split_by_h2(text)
    for title, body in segments:
        yield OperatorDefaultsSubsection(
            title=title,
            anchor=_slugify(title),
            body_markdown=body.strip(),
        )


# ---------------------------------------------------------------------------
# Slug helper
# ---------------------------------------------------------------------------


def _slugify(text: str) -> str:
    """Lowercase, dash-separated, alphanumeric-only slug for anchors."""
    lowered = text.lower()
    dashed = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    return dashed


__all__ = [
    "KubernetesEnumValue",
    "KubernetesField",
    "KubernetesPackage",
    "KubernetesReference",
    "KubernetesType",
    "KubernetesTypeRef",
    "OperatorDefaults",
    "OperatorDefaultsSubsection",
    "parse_reference",
]
