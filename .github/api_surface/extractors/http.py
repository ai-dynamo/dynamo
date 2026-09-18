# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP + env + config surface extractor.

This module owns three surfaces that share the same source tree, so they are
extracted together instead of duplicated across three near-identical scanners:

- ``http`` -- request-routing endpoints. The primary source is the
  ``RouteDoc::new(<Method>, <path>)`` call sites in the Rust HTTP service
  (``lib/llm/src/http/service/*.rs`` in ai-dynamo) -- Dynamo's own
  route-documentation constructor, which is the single thing every public
  endpoint is registered through and the same data that drives the generated
  OpenAPI. If a built ``<repo_path>/openapi.json`` is also present it is parsed
  and *preferred* per endpoint (it carries request/response property
  fingerprints the route sites cannot); the two are unioned by id. Parsing
  routes from source means the surface is covered at every release ref without
  a build step.
- ``env`` -- ``DYN_*`` environment-variable string literals in Rust / Python
  source.
- ``config`` -- ``clap::Parser`` / ``serde::Deserialize`` struct fields in Rust
  source.

Each emitted ``SurfaceSymbol`` carries its OWN ``surface`` field ("http",
"env", or "config"), so the rest of the engine sees them as three independent
surfaces even though one module produced them. The module-level ``SURFACE``
constant reports the primary surface ("http"); ``metadata["surfaces"]`` lists
all three.

The coverage map is per-source: ``coverage["http"]`` is True iff at least one
``RouteDoc::new`` site resolved OR ``openapi.json`` parsed; ``coverage["env"]``
/ ``coverage["config"]`` are True iff at least one source root was found and
walked. ``covered`` is True if ANY of the three landed -- a missing source
appends an ``OpError`` to ``errors`` rather than raising, so the diff engine
records a coverage gap and never reads the absence as a wave of removals.

Heuristics (best-effort, regex / line scan only -- no real parser):

- **ENV** -- a quoted ``"DYN_[A-Z0-9_]+"`` substring inside a ``.rs`` / ``.py``
  file. Match is on the string literal so ad-hoc identifiers like
  ``let dyn_foo = ...`` do not trip the scan; doc comments mentioning the
  exact ``"DYN_FOO"`` literal will (acceptable for best-effort).
- **CONFIG** -- a ``#[derive(...)]`` whose trait list contains ``Parser``,
  followed by ``struct <Name> {``. Fields are the field-shaped lines inside the
  matched ``{ ... }`` body (nested brace blocks are stripped first so inline
  default expressions / impls do not leak in). Tuple structs and unit structs
  are skipped. Classification is by derive, NOT by path: a ``clap::Parser``
  struct is operational config (CLI args) wherever it lives, while a struct
  that only derives serde (``Deserialize`` / ``Serialize``) is a wire /
  serialization contract (OpenAI / Anthropic request-response shapes, internal
  message types) and is excluded everywhere. Keying on the derive rather than
  the directory keeps a wire type's classification stable when it moves crates
  (e.g. ``ApiError`` migrating out of ``async-openai`` into ``protocols/``), so
  a relocation no longer reads as a phantom config removal.

Bounded scanning: only files under a small set of default roots are read;
files larger than 1 MiB and any non-``.rs`` / non-``.py`` files are skipped;
text decoding falls back to ``errors="replace"`` so a stray binary blob in a
``src/`` tree cannot crash the scan.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from api_surface.models import SurfaceSymbol
from api_surface.results import OperationResult, OpError

SURFACE = "http"
_OWNED_SURFACES: tuple[str, ...] = ("http", "env", "config")

_MAX_FILE_BYTES = 1_048_576
_SOURCE_SUFFIXES = frozenset({".rs", ".py"})
_RUST_SUFFIXES = frozenset({".rs"})

# Default source roots scanned for env/config. Existence-gated; if none are
# present we fall back to scanning ``repo_path`` directly (covers tiny test
# fixtures that put a single ``src/`` next to ``openapi.json``).
_DEFAULT_SOURCE_ROOTS: tuple[str, ...] = (
    "src",
    "lib",
    "components",
    "launch",
    "deploy",
)

_HTTP_METHODS = frozenset({"get", "post", "put", "patch", "delete", "head", "options"})

# --- Rust route parsing (RouteDoc::new) ----------------------------------
# Dynamo registers every documented endpoint through
# ``RouteDoc::new(<method_expr>, <path_expr>)``. Neither argument contains a
# comma in practice (method is ``Method::POST``-shaped, path is a literal or a
# bare identifier / ``&identifier``), so a two-group capture is sufficient.
_ROUTEDOC = re.compile(r"RouteDoc::new\(\s*([^,]+?)\s*,\s*([^)]+?)\s*\)")
_METHOD_VARIANT = re.compile(r"Method::(\w+)")
# A single-line ``let <ident> = <expr>;`` binding (path vars are always one line).
_LET = re.compile(r"^\s*let\s+(\w+)\s*=\s*(.+?);\s*$")
# A bare double-quoted Rust string literal, escaped chars tolerated.
_STR_LITERAL = re.compile(r'"((?:[^"\\]|\\.)*)"')
_UNWRAP_CALL = re.compile(r"unwrap_or(?:_else)?\s*\(")
_FORMAT_CALL = re.compile(r'^format!\(\s*"((?:[^"\\]|\\.)*)"\s*,\s*(.+)\)$')
_REPLACE_CALL = re.compile(
    r'^(.+?)\.replace\(\s*"((?:[^"\\]|\\.)*)"\s*,\s*"((?:[^"\\]|\\.)*)"\s*\)$'
)
_BARE_IDENT = re.compile(r"\w+")
# Trailing string-conversion calls that wrap a literal (``"x".to_string()``).
_STR_CONV_SUFFIX = re.compile(r"\.(?:to_string|to_owned|into)\(\)\s*$")
# Guard against pathological mutual / self references in let-chains.
_MAX_RESOLVE_DEPTH = 16

# Match a quoted DYN_* token in either single or double quotes; the bracketing
# quotes do not have to match, which is fine because malformed mixed-quote
# strings are not real env-var references.
_ENV_PATTERN = re.compile(r'["\'](DYN_[A-Z0-9_]+)["\']')

# Locate every `#[derive(...)] ... struct <Name> {` block. The trait list goes
# in `traits`; the struct name in `name`. The body extends from `m.end()` (just
# past the opening brace) to its matching `}`.
_STRUCT_HEADER = re.compile(
    r"#\[derive\((?P<traits>[^)]*)\)\]"
    r"(?:\s*#\[[^\]]*\])*"
    r"\s*(?:pub(?:\([^)]*\))?\s+)?struct\s+(?P<name>\w+)"
    r"(?:\s*<[^>]*>)?"
    r"\s*\{",
)

# A single-line field declaration: `[pub] field_name: <type>[,]`.
_FIELD_LINE = re.compile(
    r"^\s*(?:pub(?:\([^)]*\))?\s+)?"
    r"(?P<field>[A-Za-z_][A-Za-z0-9_]*)\s*:\s*"
    r"(?P<type>[^,]+?)\s*,?\s*$",
)


# =============================================================================
# Public entry point
# =============================================================================


def extract(repo_path: Path, release: str) -> OperationResult:
    """Extract http / env / config surfaces from ``repo_path`` at ``release``.

    Returns a single ``OperationResult`` whose ``data["symbols"]`` mixes all
    three surfaces. Each symbol's ``surface`` field identifies which one it
    belongs to. ``metadata["coverage"]`` reports per-source coverage so the
    diff engine never reads a missing extractor as a removal.
    """
    repo_path = Path(repo_path)
    symbols: list[SurfaceSymbol] = []
    errors: list[OpError] = []
    coverage: dict[str, bool] = {"http": False, "env": False, "config": False}

    roots = _discover_source_roots(repo_path)

    http_symbols, http_covered, http_schema_covered = _extract_http(repo_path, roots)
    if http_covered:
        coverage["http"] = True
        symbols.extend(http_symbols)
    else:
        errors.append(
            OpError(
                operation="extract.http",
                message="no http surface found (no RouteDoc::new sites and no openapi.json)",
                details={"repo_path": str(repo_path)},
            )
        )

    if not roots:
        errors.append(
            OpError(
                operation="extract.env",
                message="no source roots found under repo_path",
                details={"repo_path": str(repo_path)},
            )
        )
        errors.append(
            OpError(
                operation="extract.config",
                message="no source roots found under repo_path",
                details={"repo_path": str(repo_path)},
            )
        )
    else:
        symbols.extend(_extract_env(roots, repo_path))
        coverage["env"] = True
        symbols.extend(_extract_config(roots, repo_path))
        coverage["config"] = True

    return OperationResult(
        data={"symbols": [s.to_dict() for s in symbols]},
        errors=errors,
        metadata={
            "surface": SURFACE,
            "surfaces": list(_OWNED_SURFACES),
            "coverage": coverage,
            "coverage_detail": {"http:schema": http_schema_covered},
            "covered": any(coverage.values()),
            "release": release,
            "repo_path": str(repo_path),
            "symbol_count": len(symbols),
        },
    )


# =============================================================================
# HTTP (Rust RouteDoc routes, with optional OpenAPI enrichment)
# =============================================================================


def _extract_http(
    repo_path: Path, roots: list[Path]
) -> tuple[list[SurfaceSymbol], bool, bool]:
    """Build the http endpoint surface, unioned by id across two sources.

    The route sites in the Rust HTTP service are the always-available primary
    source. A built ``openapi.json`` (rare in source trees) is *preferred* per
    endpoint when present because it carries request/response property
    fingerprints the route sites cannot. Union is by id, openapi last so it
    wins on collision.

    Returns ``(symbols, covered, schema_covered)``. ``covered`` is true iff
    either source was found and parsed; ``schema_covered`` is true only when a
    generated OpenAPI document supplied request/response contract detail.
    """
    by_id: dict[str, SurfaceSymbol] = {}
    route_symbols, routes_present = _extract_routes(roots, repo_path)
    for sym in route_symbols:
        by_id[sym.id] = sym
    openapi_symbols, openapi_present = _extract_openapi(repo_path)
    for sym in openapi_symbols:
        by_id[sym.id] = sym
    return (
        sorted(by_id.values(), key=lambda s: s.id),
        routes_present or openapi_present,
        openapi_present,
    )


def _extract_openapi(repo_path: Path) -> tuple[list[SurfaceSymbol], bool]:
    """Parse ``repo_path/openapi.json`` into ``(symbols, present)``.

    Absence is normal -- the spec is a build artifact -- so this never errors;
    ``present`` is True iff the file existed and parsed to a dict with a
    ``paths`` map (even an empty one). The Rust route parser is the primary
    source and coverage is judged on the union.
    """
    openapi_path = repo_path / "openapi.json"
    if not openapi_path.is_file():
        return [], False
    try:
        spec = json.loads(openapi_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return [], False

    paths = spec.get("paths") if isinstance(spec, dict) else None
    if not isinstance(paths, dict):
        return [], False

    symbols: list[SurfaceSymbol] = []
    for path, methods in sorted(paths.items()):
        if not isinstance(methods, dict):
            continue
        for method, operation in sorted(methods.items()):
            if method.lower() not in _HTTP_METHODS or not isinstance(operation, dict):
                continue
            signature, metadata = _http_signature(operation)
            symbols.append(
                SurfaceSymbol(
                    surface="http",
                    kind="endpoint",
                    id=f"http:{method.upper()} {path}",
                    signature=signature,
                    deprecated=bool(operation.get("deprecated", False)),
                    deprecated_note=(
                        str(operation.get("description", ""))
                        if operation.get("deprecated", False)
                        else ""
                    ),
                    metadata=metadata,
                )
            )
    return symbols, True


# --- Rust route parsing ------------------------------------------------------


def _extract_routes(
    roots: list[Path], repo_path: Path
) -> tuple[list[SurfaceSymbol], bool]:
    """Emit ``(symbols, present)`` from ``RouteDoc::new`` sites under ``roots``.

    Only ``.rs`` files containing the literal ``RouteDoc::new(`` are parsed (a
    cheap substring gate confines the work to the HTTP service). ``present`` is
    True iff at least one such file was found -- so a route source that yields
    no statically resolvable endpoint still counts as covered, distinguishing
    "parsed, zero endpoints" from "no http source at all". Endpoints whose path
    cannot be resolved to a literal are skipped rather than emitted with a
    placeholder, so the surface never carries a synthetic id. Deduplicated by
    id (e.g. the OpenAI and Anthropic routers both default model listing to
    ``/v1/models``).
    """
    by_id: dict[str, SurfaceSymbol] = {}
    present = False
    for path in _iter_source_files(roots, _RUST_SUFFIXES):
        text = _safe_read(path)
        if "RouteDoc::new(" not in text:
            continue
        present = True
        for sym in _scan_routes(text, path, repo_path):
            by_id.setdefault(sym.id, sym)
    return sorted(by_id.values(), key=lambda s: s.id), present


def _scan_routes(text: str, path: Path, repo_path: Path) -> list[SurfaceSymbol]:
    """Yield endpoint symbols from one Rust source file's ``RouteDoc::new`` sites."""
    lets = _collect_lets(text)
    out: list[SurfaceSymbol] = []
    for match in _ROUTEDOC.finditer(text):
        method_match = _METHOD_VARIANT.search(match.group(1))
        if method_match is None:
            continue
        method = method_match.group(1).upper()
        if method.lower() not in _HTTP_METHODS:
            continue
        line_idx = text.count("\n", 0, match.start())
        resolved = _resolve_path_arg(match.group(2).strip(), line_idx, lets)
        if resolved is None:
            continue
        out.append(
            SurfaceSymbol(
                surface="http",
                kind="endpoint",
                id=f"http:{method} {resolved}",
                signature="",
                metadata={
                    "method": method,
                    "path": resolved,
                    "source": "rust_route",
                    "source_file": _relative_path(path, repo_path),
                    "confidence": "high",
                },
            )
        )
    return out


def _collect_lets(text: str) -> list[tuple[int, str, str]]:
    """Index every single-line ``let <ident> = <expr>;`` as ``(line_idx, ident, expr)``."""
    lets: list[tuple[int, str, str]] = []
    for i, line in enumerate(text.splitlines()):
        m = _LET.match(line)
        if m is not None:
            lets.append((i, m.group(1), m.group(2).strip()))
    return lets


def _resolve_path_arg(
    arg: str, line_idx: int, lets: list[tuple[int, str, str]]
) -> str | None:
    """Resolve a ``RouteDoc::new`` path argument to a literal path, or ``None``."""
    return _resolve_expr(arg.lstrip("&").strip(), line_idx, lets, 0)


def _resolve_var(
    var: str, before_idx: int, lets: list[tuple[int, str, str]], depth: int
) -> str | None:
    """Resolve ``var`` from the nearest ``let var = ...`` strictly above ``before_idx``.

    Nearest-preceding binding (not first / last) is what makes the function-local
    scoping correct: a name like ``path`` is rebound in every router fn, and each
    ``RouteDoc::new`` sees its own function's binding.
    """
    if depth > _MAX_RESOLVE_DEPTH:
        return None
    best: tuple[int, str] | None = None
    for idx, name, expr in lets:
        if name == var and idx < before_idx and (best is None or idx > best[0]):
            best = (idx, expr)
    if best is None:
        return None
    return _resolve_expr(best[1], best[0], lets, depth + 1)


def _resolve_expr(
    expr: str, at_idx: int, lets: list[tuple[int, str, str]], depth: int
) -> str | None:
    """Resolve a path-building expression to a literal string, or ``None``.

    Handles the closed set of shapes the HTTP service uses: a direct string
    literal (with optional ``.to_string()``), an ``unwrap_or[_else]`` default,
    ``format!("{}<suffix>", base)``, ``base.replace("a", "b")``, and a bare
    identifier (resolved against ``lets``). Anything else returns ``None`` and
    the endpoint is skipped.
    """
    if depth > _MAX_RESOLVE_DEPTH:
        return None
    expr = expr.strip()

    literal = _as_literal(expr)
    if literal is not None:
        return literal

    if _UNWRAP_CALL.search(expr):
        m = _STR_LITERAL.search(expr)
        return _unescape(m.group(1)) if m is not None else None

    fmt = _FORMAT_CALL.match(expr)
    if fmt is not None:
        base = _resolve_expr(fmt.group(2).lstrip("&").strip(), at_idx, lets, depth + 1)
        return _apply_format(fmt.group(1), base) if base is not None else None

    rep = _REPLACE_CALL.match(expr)
    if rep is not None:
        base = _resolve_expr(rep.group(1).lstrip("&").strip(), at_idx, lets, depth + 1)
        if base is None:
            return None
        return base.replace(_unescape(rep.group(2)), _unescape(rep.group(3)))

    if _BARE_IDENT.fullmatch(expr):
        return _resolve_var(expr, at_idx, lets, depth + 1)

    return None


def _as_literal(expr: str) -> str | None:
    """Return the string value of a literal expression (``"x"`` / ``"x".to_string()``)."""
    stripped = _STR_CONV_SUFFIX.sub("", expr.strip()).strip()
    m = _STR_LITERAL.fullmatch(stripped)
    return _unescape(m.group(1)) if m is not None else None


def _apply_format(template: str, base: str) -> str:
    """Substitute ``base`` into the first ``{}`` of a ``format!`` template.

    The path variable is always the first positional placeholder; after
    substitution the Rust brace escapes ``{{`` / ``}}`` are lowered to literal
    ``{`` / ``}`` (e.g. ``"{}/{{*model_id}}"`` + ``/v1/models`` ->
    ``/v1/models/{*model_id}``).
    """
    return template.replace("{}", base, 1).replace("{{", "{").replace("}}", "}")


def _unescape(value: str) -> str:
    """Lower the escape sequences that appear in Rust path literals (``\\"`` / ``\\\\``)."""
    return value.replace('\\"', '"').replace("\\\\", "\\")


def _http_signature(operation: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Build a stable request/response property fingerprint for one operation.

    The signature joins canonical request properties and 2xx response schemas.
    Property types, constraints, requiredness, and raw ``$ref`` values are part
    of the signature so an incompatible contract change reaches the diff gate.
    """
    req_props, req_ref = _schema_properties(operation.get("requestBody"))
    resp_props, resp_refs = _response_properties(operation.get("responses"))
    signature = f"req({','.join(req_props)}) -> resp({','.join(resp_props)})"
    metadata: dict[str, Any] = {}
    if req_ref:
        metadata["request_ref"] = req_ref
    if resp_refs:
        metadata["response_refs"] = resp_refs
    return signature, metadata


_SCHEMA_CONTRACT_KEYS = (
    "$ref",
    "type",
    "format",
    "enum",
    "const",
    "nullable",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "minLength",
    "maxLength",
    "pattern",
    "minItems",
    "maxItems",
    "uniqueItems",
    "minProperties",
    "maxProperties",
    "additionalProperties",
)


def _schema_contract(schema: dict[str, Any]) -> str:
    """Canonical JSON projection of compatibility-relevant schema fields."""
    contract = {key: schema[key] for key in _SCHEMA_CONTRACT_KEYS if key in schema}
    required = schema.get("required")
    if isinstance(required, list):
        contract["required"] = sorted(str(item) for item in required)
    properties = schema.get("properties")
    if isinstance(properties, dict):
        contract["properties"] = {
            str(name): _schema_contract(value)
            for name, value in sorted(properties.items())
            if isinstance(value, dict)
        }
    items = schema.get("items")
    if isinstance(items, dict):
        contract["items"] = _schema_contract(items)
    for key in ("allOf", "anyOf", "oneOf"):
        variants = schema.get(key)
        if isinstance(variants, list):
            contract[key] = [
                _schema_contract(item) for item in variants if isinstance(item, dict)
            ]
    return json.dumps(contract, sort_keys=True, separators=(",", ":"))


def _schema_properties(body: Any) -> tuple[list[str], str]:
    """Canonical schema fingerprint + top-level $ref from a content wrapper."""
    if not isinstance(body, dict):
        return [], ""
    content = body.get("content")
    if not isinstance(content, dict):
        return [], ""
    media = content.get("application/json")
    if not isinstance(media, dict):
        for value in content.values():
            if isinstance(value, dict):
                media = value
                break
        else:
            return [], ""
    schema = media.get("schema")
    if not isinstance(schema, dict):
        return [], ""
    ref = schema.get("$ref") if isinstance(schema.get("$ref"), str) else ""
    return [_schema_contract(schema)], ref or ""


def _response_properties(responses: Any) -> tuple[list[str], dict[str, str]]:
    """Collect 2xx response property names + $refs keyed by status code."""
    if not isinstance(responses, dict):
        return [], {}
    refs: dict[str, str] = {}
    seen: set[str] = set()
    for status in sorted(responses):
        if not str(status).startswith("2"):
            continue
        names, ref = _schema_properties(responses[status])
        seen.update(f"{status}:{name}" for name in names)
        if ref:
            refs[str(status)] = ref
    return sorted(seen), refs


# =============================================================================
# ENV (DYN_* scanning)
# =============================================================================


def _relative_path(path: Path, repo_path: Path) -> str:
    """Return a portable repository-relative POSIX source path."""
    try:
        return path.relative_to(repo_path).as_posix()
    except ValueError:
        return path.as_posix()


def _extract_env(roots: list[Path], repo_path: Path) -> list[SurfaceSymbol]:
    """Find every quoted ``DYN_*`` literal under ``roots`` and emit one symbol per name."""
    sightings: dict[str, set[str]] = {}
    for path in _iter_source_files(roots, _SOURCE_SUFFIXES):
        text = _safe_read(path)
        if "DYN_" not in text:
            continue
        for name in set(_ENV_PATTERN.findall(text)):
            sightings.setdefault(name, set()).add(_relative_path(path, repo_path))
    symbols: list[SurfaceSymbol] = []
    for name in sorted(sightings):
        symbols.append(
            SurfaceSymbol(
                surface="env",
                kind="env_var",
                id=f"env:{name}",
                signature="",
                metadata={"source_files": sorted(sightings[name])},
            )
        )
    return symbols


# =============================================================================
# CONFIG (clap::Parser / serde::Deserialize struct fields)
# =============================================================================


def _extract_config(roots: list[Path], repo_path: Path) -> list[SurfaceSymbol]:
    """Find clap config fields under ``roots``, deduplicated by stable id."""
    by_id: dict[str, SurfaceSymbol] = {}
    for path in _iter_source_files(roots, _RUST_SUFFIXES):
        text = _safe_read(path)
        if "#[derive(" not in text:
            continue
        for sym in _scan_config_structs(text, path, repo_path):
            by_id.setdefault(sym.id, sym)
    return sorted(by_id.values(), key=lambda s: s.id)


def _scan_config_structs(text: str, path: Path, repo_path: Path) -> list[SurfaceSymbol]:
    """Yield config field symbols from a single Rust source file.

    Operational config is ``clap::Parser`` or flattened ``clap::Args``. A struct that only
    derives serde (``Deserialize`` / ``Serialize``) is a wire / serialization
    contract, not config, and is skipped regardless of where it lives -- so a
    wire type keeps its classification when it moves crates and a relocation is
    never misread as a config removal.
    """
    out: list[SurfaceSymbol] = []
    source_file = _relative_path(path, repo_path)
    source_id = Path(source_file).with_suffix("").as_posix()
    for match in _STRUCT_HEADER.finditer(text):
        traits = {
            item.strip().split("::")[-1] for item in match.group("traits").split(",")
        }
        if not traits.intersection({"Parser", "Args"}):
            continue
        struct_name = match.group("name")
        body = _braced_body(text, match.end())
        if body is None:
            continue
        flat = _strip_nested_braces(body)
        pending_attributes: list[str] = []
        for line in flat.splitlines():
            stripped = line.strip()
            if stripped.startswith("#["):
                pending_attributes.append(stripped)
                continue
            field = _parse_field_line(line)
            if field is None:
                if stripped and not stripped.startswith("///"):
                    pending_attributes.clear()
                continue
            fname, ftype = field
            attributes = " ".join(pending_attributes)
            hidden = bool(re.search(r"\bhide\s*=\s*true\b", attributes))
            structural = bool(re.search(r"\b(flatten|subcommand)\b", attributes))
            skipped = bool(re.search(r"\bskip\b(?!\s*=\s*false\b)", attributes))
            pending_attributes.clear()
            if hidden or structural or skipped:
                continue
            out.append(
                SurfaceSymbol(
                    surface="config",
                    kind="field",
                    id=f"config:{source_id}::{struct_name}.{fname}",
                    signature=ftype,
                    metadata={"source_file": source_file, "struct": struct_name},
                )
            )
    return out


def _braced_body(text: str, body_start: int) -> str | None:
    """Return the substring inside the ``{...}`` that begins at ``body_start``.

    ``body_start`` is the position immediately AFTER the opening ``{``. Returns
    ``None`` if the braces are not balanced (truncated source).
    """
    depth = 1
    idx = body_start
    n = len(text)
    while idx < n and depth > 0:
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        idx += 1
    if depth != 0:
        return None
    return text[body_start : idx - 1]


def _strip_nested_braces(body: str) -> str:
    """Remove any ``{...}`` spans so field-line regex doesn't match inner content.

    This drops things like ``#[serde(default = || { ... })]`` blocks and any
    accidental inline ``impl`` bodies, keeping only depth-zero content.
    """
    out: list[str] = []
    depth = 0
    for ch in body:
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            if depth > 0:
                depth -= 1
            continue
        if depth == 0:
            out.append(ch)
    return "".join(out)


def _parse_field_line(line: str) -> tuple[str, str] | None:
    """Match a single struct-field line, skipping attributes / comments / blanks."""
    stripped = line.strip()
    if not stripped:
        return None
    if stripped.startswith(("//", "#", "/*", "*/", "*")):
        return None
    match = _FIELD_LINE.match(line)
    if match is None:
        return None
    return match.group("field"), match.group("type").strip()


# =============================================================================
# Filesystem helpers
# =============================================================================


def _discover_source_roots(repo_path: Path) -> list[Path]:
    """Return the default source roots that exist under ``repo_path``.

    Falls back to ``[repo_path]`` itself when none of the defaults are
    present but the repo still has scannable source files (covers the tiny
    test fixture layout).
    """
    roots = [
        repo_path / name
        for name in _DEFAULT_SOURCE_ROOTS
        if (repo_path / name).is_dir()
    ]
    if roots:
        return roots
    if not repo_path.is_dir():
        return []
    # Only adopt the repo root if it actually contains scannable source files
    # (cheap: a single peek aborts the walk).
    for _ in _iter_source_files([repo_path], _SOURCE_SUFFIXES):
        return [repo_path]
    return []


def _iter_source_files(
    roots: Iterable[Path], suffixes: frozenset[str]
) -> Iterator[Path]:
    """Yield text files under ``roots`` whose suffix is in ``suffixes``.

    Skips files larger than ``_MAX_FILE_BYTES`` and silently drops any path
    whose ``stat()`` fails (broken symlinks, races during a build).
    """
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in suffixes:
                continue
            try:
                if path.stat().st_size > _MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            yield path


def _safe_read(path: Path) -> str:
    """Read ``path`` as UTF-8 text, replacing undecodable bytes."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
