# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit project annotations for tiers and non-native deprecation signals."""

from __future__ import annotations

from pathlib import Path

import yaml

from .models import STABILITY_TIERS, SurfaceSymbol
from .results import OperationResult, OpError

_ALLOWED_FIELDS = {"stability", "deprecated", "note", "removal_target"}


def _validate_annotation(symbol_id: str, annotation: dict[object, object]) -> str:
    """Return an error message for one annotation, or an empty string."""
    unknown = sorted(str(key) for key in annotation if key not in _ALLOWED_FIELDS)
    if unknown:
        return f"unknown annotation field(s) for {symbol_id}: {', '.join(unknown)}"
    stability = annotation.get("stability")
    if stability is not None and (
        not isinstance(stability, str) or stability not in STABILITY_TIERS
    ):
        return f"invalid stability tier for {symbol_id}: {stability}"
    deprecated = annotation.get("deprecated")
    if deprecated is not None and not isinstance(deprecated, bool):
        return f"deprecated must be a boolean for {symbol_id}"
    for field in ("note", "removal_target"):
        value = annotation.get(field)
        if value is not None and not isinstance(value, str):
            return f"{field} must be a string for {symbol_id}"
    return ""


def _apply_annotation(symbol: SurfaceSymbol, annotation: dict[object, object]) -> None:
    """Apply one already-validated annotation without weakening native markers."""
    stability = annotation.get("stability")
    if isinstance(stability, str):
        symbol.stability = stability
    if annotation.get("deprecated") is True:
        symbol.deprecated = True
    note = annotation.get("note")
    if isinstance(note, str) and note:
        symbol.deprecated_note = note
    removal_target = annotation.get("removal_target")
    if isinstance(removal_target, str) and removal_target:
        symbol.metadata["removal_target"] = removal_target


def default_annotations_path(root: str | Path = ".") -> Path:
    """Return the project-local API annotation manifest path."""
    return Path(root) / ".github" / "api-surface" / "annotations.yaml"


def apply_annotations(
    symbols: list[SurfaceSymbol],
    path: str | Path,
) -> OperationResult:
    """Apply explicit tier and deprecation annotations to extracted symbols.

    The manifest uses a ``symbols`` mapping keyed by the tracker's stable symbol
    IDs. Supported fields are ``stability``, ``deprecated``, ``note``, and
    ``removal_target``. Invalid tiers and unknown IDs are errors so a typo can
    never silently weaken the stability contract.
    """
    annotation_path = Path(path)
    result = OperationResult(
        metadata={"path": str(annotation_path), "present": annotation_path.is_file()}
    )
    if not annotation_path.is_file():
        return result
    try:
        document = yaml.safe_load(annotation_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as error:
        result.errors.append(
            OpError(
                operation="api_surface.annotations",
                message=f"failed to load {annotation_path}: {error}",
            )
        )
        return result

    raw = document.get("symbols") if isinstance(document, dict) else None
    if not isinstance(raw, dict):
        result.errors.append(
            OpError(
                operation="api_surface.annotations",
                message="annotation manifest must contain a symbols mapping",
            )
        )
        return result

    index = {symbol.id: symbol for symbol in symbols}
    applied = 0
    for symbol_id, annotation in raw.items():
        if not isinstance(symbol_id, str) or not isinstance(annotation, dict):
            result.errors.append(
                OpError(
                    operation="api_surface.annotations",
                    message="every annotation must map a symbol ID to a mapping",
                )
            )
            continue
        symbol = index.get(symbol_id)
        if symbol is None:
            result.errors.append(
                OpError(
                    operation="api_surface.annotations",
                    message=f"annotation references unknown symbol: {symbol_id}",
                )
            )
            continue
        validation_error = _validate_annotation(symbol_id, annotation)
        if validation_error:
            result.errors.append(
                OpError(
                    operation="api_surface.annotations",
                    message=validation_error,
                )
            )
            continue
        _apply_annotation(symbol, annotation)
        applied += 1

    result.metadata["applied"] = applied
    return result
