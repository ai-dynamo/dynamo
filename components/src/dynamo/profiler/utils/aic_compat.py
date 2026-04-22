# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility helpers for AIConfigurator integration in the profiler."""

from __future__ import annotations

import contextlib
import copy
import importlib
import logging
from collections.abc import Mapping
from typing import Any

logger = logging.getLogger(__name__)

_BLACKWELL_SYSTEMS = {"b200_sxm", "gb200_sxm"}
_MXFP4_MODE_ALIASES = {
    "w4a16_mxfp4": "nvfp4",
}
_MODEL_CONFIG_ATTRS = (
    ("aiconfigurator.sdk.utils", "get_model_config_from_model_path"),
    ("aiconfigurator.sdk.task", "get_model_config_from_model_path"),
    ("aiconfigurator.generator.enumerate", "get_model_config_from_model_path"),
    ("aiconfigurator.cli.main", "get_model_config_from_model_path"),
)


def _is_blackwell_system(system_name: str | None) -> bool:
    return (system_name or "").lower() in _BLACKWELL_SYSTEMS


def normalize_aic_model_config(
    model_config: Mapping[str, Any] | Any,
    *,
    system_name: str | None,
) -> Mapping[str, Any] | Any:
    """Normalize model-config fields that older AIC builds do not accept.

    Blackwell perf tables use ``nvfp4`` while some model configs surface the
    MoE quant mode as ``w4a16_mxfp4``. Normalize that alias narrowly so AIC can
    validate Blackwell MXFP4 models without relaxing other systems.
    """
    if not _is_blackwell_system(system_name) or not isinstance(model_config, Mapping):
        return model_config

    normalized = copy.deepcopy(dict(model_config))
    changed = False
    for key in ("moe_mode", "moe_quant_mode"):
        value = normalized.get(key)
        replacement = _MXFP4_MODE_ALIASES.get(value)
        if replacement is not None:
            normalized[key] = replacement
            changed = True

    if changed:
        logger.info(
            "Normalized AIConfigurator MoE quant mode for Blackwell: %s",
            {
                key: normalized[key]
                for key in ("moe_mode", "moe_quant_mode")
                if key in normalized
            },
        )

    return normalized


@contextlib.contextmanager
def patched_aic_model_config(system_name: str | None):
    """Temporarily patch AIC model-config loading with Dynamo compatibility fixes."""
    if not _is_blackwell_system(system_name):
        yield
        return

    patched: list[tuple[object, str, Any]] = []
    try:
        for module_name, attr_name in _MODEL_CONFIG_ATTRS:
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue
            original = getattr(module, attr_name, None)
            if not callable(original):
                continue

            def _wrapper(*args, _original=original, **kwargs):
                return normalize_aic_model_config(
                    _original(*args, **kwargs),
                    system_name=system_name,
                )

            setattr(module, attr_name, _wrapper)
            patched.append((module, attr_name, original))
        yield
    finally:
        for module, attr_name, original in reversed(patched):
            setattr(module, attr_name, original)
