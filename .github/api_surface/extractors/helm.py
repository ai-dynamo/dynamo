# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helm values.yaml surface extractor.

Walks ``<repo>/deploy/helm/charts/**/values.yaml``, parses each file with
PyYAML, and emits one :class:`SurfaceSymbol` per leaf key. The chart name is
the first path segment under ``charts/`` (e.g. ``platform`` for both
``charts/platform/values.yaml`` and
``charts/platform/components/operator/values.yaml``) so a chart's surface
stays under one stable namespace regardless of how its files are nested.

A chart can ship multiple ``values.yaml`` files -- its own plus subcharts
under ``components/<name>/``. To keep ids unique (two files would otherwise
collide on common keys like ``image`` or ``resources``), the intra-chart
subdirectory is folded into the dotted key, mirroring how Helm addresses
subchart values: ``charts/platform/components/operator/values.yaml`` key
``image.repo`` becomes ``helm:platform:components.operator.image.repo``.

Signature semantics: the signature carries the value's *type*, never the
literal default. Helm chart defaults churn release-over-release without
representing an API change (e.g. bumping a tag from ``v1.4`` to ``v1.5``), so
the diff engine must compare types instead. The literal default is preserved
in ``metadata["default"]`` for renderers that want to surface it.

List handling: a list emits one symbol with signature ``list``; element
indices are not addressable since Helm list semantics (merge-by-index vs
replace) are not type-stable across releases.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import yaml
from api_surface.models import SurfaceSymbol
from api_surface.results import OperationResult, OpError

SURFACE = "helm"

_OPERATION = "helm.extract"


def _type_of(value: Any) -> str:
    """Canonical signature type for a Helm values leaf.

    ``bool`` is intentionally checked before ``int`` because Python's ``bool``
    is a subclass of ``int`` -- without the explicit check, ``True`` would
    serialize as ``int``.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "map"
    # PyYAML can also yield datetime / date scalars; treat as opaque strings.
    return "string"


def _walk(
    node: Any,
    prefix: list[str],
    out: list[tuple[list[str], Any]],
) -> None:
    """Collect (dotted-key-path, leaf-value) pairs from a parsed YAML tree.

    Non-empty mappings are recursed into; everything else (scalars, lists,
    ``None``, empty mappings) is a leaf. The diff engine never crosses into
    list elements, so lists are leaves regardless of their contents.
    """
    if isinstance(node, dict) and node:
        for key in node:
            _walk(node[key], [*prefix, str(key)], out)
        return
    out.append((prefix, node))


def _chart_name(rel_to_charts: Path) -> str:
    """Chart name = first path segment under ``charts/``.

    Returns an empty string when the file sits directly in ``charts/`` with no
    chart subdirectory; the caller skips those rather than emitting symbols
    with an empty chart namespace.
    """
    parts = rel_to_charts.parts
    if len(parts) < 2:
        return ""
    return parts[0]


def extract(repo_path: Path, release: str) -> OperationResult:
    """Extract Helm chart values from ``repo_path / deploy/helm/charts``.

    Returns an :class:`OperationResult` with:

    - ``data["symbols"]``: list of :meth:`SurfaceSymbol.to_dict` outputs
    - ``metadata["surface"]``: ``"helm"``
    - ``metadata["covered"]``: ``True`` when the charts dir exists (even if a
      single file fails to parse), ``False`` when it is absent

    Per the extractor protocol, failures append :class:`OpError` to
    ``errors`` -- this function never raises.
    """
    start = time.time()
    result = OperationResult(metadata={"surface": SURFACE, "covered": False})
    result.data["symbols"] = []

    charts_dir = repo_path / "deploy" / "helm" / "charts"
    if not charts_dir.is_dir():
        result.errors.append(
            OpError(
                operation=_OPERATION,
                message=f"Helm charts directory not found: {charts_dir}",
                details={"repo_path": str(repo_path), "release": release},
            )
        )
        result.add_timing(start)
        return result

    symbols: list[dict[str, Any]] = []
    chart_names: set[str] = set()
    for values_path in sorted(charts_dir.rglob("values.yaml")):
        rel_to_charts = values_path.relative_to(charts_dir)
        chart = _chart_name(rel_to_charts)
        if not chart:
            continue
        try:
            raw = values_path.read_text(encoding="utf-8")
        except OSError as exc:
            result.errors.append(
                OpError(
                    operation=_OPERATION,
                    message=f"Failed to read {values_path}: {exc}",
                    details={"path": str(values_path)},
                )
            )
            continue
        try:
            tree = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            result.errors.append(
                OpError(
                    operation=_OPERATION,
                    message=f"Failed to parse {values_path}: {exc}",
                    details={"path": str(values_path)},
                )
            )
            continue
        if tree is None:
            continue

        leaves: list[tuple[list[str], Any]] = []
        _walk(tree, [], leaves)
        rel_path = str(values_path.relative_to(repo_path))
        # A chart can ship several values.yaml files (its own + subcharts under
        # components/<name>/). Fold the intra-chart subdirectory into the key
        # namespace -- exactly how Helm addresses subchart values
        # (``operator.image``) -- so two files never collide on a shared key
        # name like ``image`` or ``resources``.
        subdir_parts = list(values_path.parent.relative_to(charts_dir / chart).parts)
        for key_path, value in leaves:
            if not key_path:
                # Top-level non-mapping document; no addressable key to emit.
                continue
            dotted = ".".join([*subdir_parts, *key_path])
            symbol = SurfaceSymbol(
                surface=SURFACE,
                kind="value",
                id=f"helm:{chart}:{dotted}",
                signature=_type_of(value),
                metadata={
                    "chart": chart,
                    "rel_path": rel_path,
                    "default": "" if value is None else str(value),
                },
            )
            symbols.append(symbol.to_dict())
            chart_names.add(chart)

    result.data["symbols"] = symbols
    result.metadata["covered"] = True
    result.metadata["chart_count"] = len(chart_names)
    result.metadata["release"] = release
    result.add_timing(start)
    return result
