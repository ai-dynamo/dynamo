# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for canary payload construction and detection.

Producer side: :func:`build_health_check_payload` — engines call this from
``LLMEngine.health_check_payload()`` with a BOS token and optional
backend-specific ``extras``; it merges extras into a 1-token default,
applies any ``DYN_HEALTH_CHECK_PAYLOAD`` operator override, and stamps the
``_HEALTH_CHECK`` marker.

Consumer side: :func:`is_probe` — engines call this from ``generate()`` to
detect canary requests and bypass cross-worker coordination.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any, Optional

from dynamo.health_check import HEALTH_CHECK_KEY, load_health_check_from_env

logger = logging.getLogger(__name__)

__all__ = [
    "HEALTH_CHECK_KEY",
    "bos_token_id_or",
    "build_health_check_payload",
    "is_probe",
    "parse_health_check_payload_cli",
]


def is_probe(request: Mapping[str, Any]) -> bool:
    """True when the request carries the canary marker; see module docstring."""
    return request.get(HEALTH_CHECK_KEY) is True


def build_health_check_payload(
    bos_token_id: int,
    *,
    extras: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Token-shape canary payload (default + extras + env override + marker)."""
    payload: dict[str, Any] = {
        "token_ids": [bos_token_id],
        "stop_conditions": {"max_tokens": 1, "ignore_eos": True},
        "sampling_options": {"temperature": 0.0},
    }
    if extras:
        payload.update(extras)
    return _finalize(payload)


def _finalize(payload: dict[str, Any]) -> dict[str, Any]:
    # Explicit `is None` so an intentional empty-dict env override
    # (`DYN_HEALTH_CHECK_PAYLOAD={}`) wins over the engine default
    # instead of falling through to it.
    env_override = load_health_check_from_env()
    return _stamp_marker(env_override if env_override is not None else payload)


def bos_token_id_or(tokenizer: Any, default: int = 1) -> int:
    """Walk ``tokenizer.tokenizer.bos_token_id``; return ``default`` if missing.
    Callers pass the backend-specific outer wrapper around the HF tokenizer."""
    inner = getattr(tokenizer, "tokenizer", None)
    bos = getattr(inner, "bos_token_id", None)
    return int(bos) if bos is not None else default


def parse_health_check_payload_cli(value: Optional[str]) -> Optional[dict[str, Any]]:
    """Parse a ``--health-check-payload`` value (JSON or ``@/path``) into a dict.

    Invalid input logs a warning and returns ``None`` so the engine default
    still applies. The ``_HEALTH_CHECK`` marker is stamped on success.
    """
    if not value:
        return None
    try:
        if value.startswith("@"):
            with open(value[1:], "r") as f:
                parsed = json.load(f)
        else:
            parsed = json.loads(value)
    except (json.JSONDecodeError, FileNotFoundError, OSError) as e:
        logger.warning("Failed to parse --health-check-payload value: %s", e)
        return None
    if not isinstance(parsed, dict):
        logger.warning(
            "--health-check-payload must be a JSON object; got %s",
            type(parsed).__name__,
        )
        return None
    return _stamp_marker(parsed)


def _stamp_marker(payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    out[HEALTH_CHECK_KEY] = True
    return out
