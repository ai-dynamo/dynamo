# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prometheus metric-name extractor (slot B4).

Parses the centralized metric-name registry at
``lib/runtime/src/metrics/prometheus_names.rs`` -- the documented single source
of truth for Dynamo's Prometheus metric names -- and emits one
:class:`SurfaceSymbol` per metric-name constant. The file is read as text and
scanned with comment-aware brace matching; nothing is compiled or executed, so
this is safe to drive against arbitrary release refs.

Why the name registry, not the registration call sites
-------------------------------------------------------
Metric *types* (counter / gauge / histogram) and some runtime *labels* live at the
``create_counter`` / ``create_gauge`` / ... call sites, scattered across ~20
files, and the full Prometheus name is composed at runtime as
``{prefix}_{suffix}`` (prefix from the ``name_prefix`` module, suffix from these
constants). Reconstructing all of that statically is brittle and call-shape
dependent. The name registry, by contrast, is one well-formed file, present at
every release tag, and is the contract that changes when a metric is added /
renamed / removed -- so it is the deterministic, diff-able surface this tracker
needs. (The earlier implementation parsed an empty ``prometheus_metrics.pyi``
type-stub and reported a vacuous ``covered=True`` with zero symbols.)

Scope and confidence (medium)
-----------------------------
We emit module-qualified metric-name and label-name constants with the string
value as the signature. We do NOT compose the runtime prefix or capture metric
types and call-site-only labels. We skip prefix / component-name modules
(:data:`_NON_METRIC_MODULES`), nested label-value sub-modules, and ``*_ENV`` /
``DYN_*`` constants (which belong to the env surface). Constants in ``labels``
or named ``*_LABEL`` are classified as ``kind="label"``.

Symbol shape
------------
For each detected metric-name constant we emit::

    SurfaceSymbol(
        surface="metric",
        kind="metric",
        id=f"metric:{module}::{const_name}",
        signature=value,                 # the name string, e.g. "requests_total"
        metadata={"name": value, "module": module, "confidence": "medium"},
    )

Keying the id on the (module, constant-name) pair keeps identity stable across a
*value* rename (which surfaces as a ``signature_changed`` diff), while an added
or deleted constant surfaces as ``added`` / ``removed``.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from api_surface.models import SurfaceSymbol
from api_surface.results import OperationResult, OpError

SURFACE = "metric"

_RELATIVE_SOURCE = Path("lib/runtime/src/metrics/prometheus_names.rs")

# Top-level modules that hold prefixes or component-name values rather than
# metric/label contract names.
_NON_METRIC_MODULES = {"name_prefix", "component_names"}

# A top-level ``pub mod NAME {`` (column 0 -- nested modules are indented and so
# do not match, which is exactly what we want: nested modules are label-value
# enums whose constants must not be emitted as metrics).
_TOP_MODULE = re.compile(r"^pub mod (\w+)\s*\{", re.MULTILINE)

# A nested ``pub mod`` opener anywhere in a module body (indented in source).
_NESTED_MODULE = re.compile(r"\bpub mod \w+\s*\{")

# ``pub const IDENT: &str = "VALUE"`` (optional lifetime tolerated).
_CONST = re.compile(r"""pub const (\w+)\s*:\s*&(?:'\w+\s+)?str\s*=\s*"([^"]*)\"""")


def _strip_comments(text: str) -> str:
    """Remove ``/* */`` blocks and ``//`` line comments, preserving newlines.

    Metric-name string values contain no ``//`` or ``/*`` sequences, so this is
    lossless for the data we extract and removes the only source of stray braces
    (doc comments such as ``{prefix}_{suffix}``) that would otherwise confuse
    brace matching.
    """
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", text)


def _matching_brace(text: str, open_idx: int) -> int:
    """Index of the ``}`` that closes the ``{`` at ``open_idx`` (-1 if none).

    Operates on comment-stripped text whose only braces are block delimiters.
    """
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _nested_spans(body: str) -> list[tuple[int, int]]:
    """Half-open (start, end) spans of nested ``pub mod`` blocks within a body."""
    spans: list[tuple[int, int]] = []
    for m in _NESTED_MODULE.finditer(body):
        open_idx = body.index("{", m.start())
        close_idx = _matching_brace(body, open_idx)
        if close_idx != -1:
            spans.append((m.start(), close_idx + 1))
    return spans


def _is_env_const(name: str, value: str) -> bool:
    """True for environment-variable constants (env surface, not metric)."""
    return name.endswith("_ENV") or value.startswith("DYN_")


def _module_metrics(module: str, body: str) -> list[SurfaceSymbol]:
    """Emit one symbol per direct metric-name constant in a module body."""
    nested = _nested_spans(body)
    symbols: list[SurfaceSymbol] = []
    for m in _CONST.finditer(body):
        if any(start <= m.start() < end for start, end in nested):
            continue
        name, value = m.group(1), m.group(2)
        if _is_env_const(name, value):
            continue
        kind = "label" if module == "labels" or name.endswith("_LABEL") else "metric"
        symbols.append(
            SurfaceSymbol(
                surface=SURFACE,
                kind=kind,
                id=f"metric:{module}::{name}",
                signature=value,
                metadata={"name": value, "module": module, "confidence": "medium"},
            )
        )
    return symbols


def _scan(text: str) -> list[SurfaceSymbol]:
    """Scan the registry text, emitting metric-name constants from metric modules."""
    stripped = _strip_comments(text)
    symbols: list[SurfaceSymbol] = []
    for m in _TOP_MODULE.finditer(stripped):
        module = m.group(1)
        if module in _NON_METRIC_MODULES:
            continue
        open_idx = stripped.index("{", m.start())
        close_idx = _matching_brace(stripped, open_idx)
        if close_idx == -1:
            continue
        body = stripped[open_idx + 1 : close_idx]
        symbols.extend(_module_metrics(module, body))
    return symbols


def extract(repo_path: Path, release: str) -> OperationResult:
    """Extract Prometheus metric-name constants from the Dynamo name registry.

    Args:
        repo_path: Repository root; the source is resolved as
            ``repo_path / "lib/runtime/src/metrics/prometheus_names.rs"``.
        release: Release version string (passed through to metadata for parity
            with the other extractors; the scan itself is release-agnostic).

    Returns:
        OperationResult with ``data["symbols"]`` (sorted by id),
        ``metadata["surface"] = "metric"``, and ``metadata["covered"]`` set to
        ``True`` when the source was read, ``False`` when it is absent or
        unreadable. Failures append :class:`OpError` rather than raising.
    """
    start = time.time()
    source_path = repo_path / _RELATIVE_SOURCE
    result = OperationResult(
        data={"symbols": []},
        metadata={
            "surface": SURFACE,
            "covered": False,
            "release": release,
            "source": str(source_path),
        },
    )

    if not source_path.is_file():
        result.errors.append(
            OpError(
                operation="extract",
                message=f"prometheus_names.rs not found at {source_path}",
                details={"surface": SURFACE, "release": release},
            )
        )
        result.add_timing(start)
        return result

    try:
        text = source_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        result.errors.append(
            OpError(
                operation="extract",
                message=f"failed to read {source_path}: {exc}",
                details={
                    "surface": SURFACE,
                    "release": release,
                    "error_type": type(exc).__name__,
                },
            )
        )
        result.add_timing(start)
        return result

    symbols = _scan(text)
    symbols.sort(key=lambda s: s.id)

    result.data["symbols"] = [s.to_dict() for s in symbols]
    result.metadata["covered"] = True
    result.metadata["symbol_count"] = len(symbols)
    result.add_timing(start)
    return result
