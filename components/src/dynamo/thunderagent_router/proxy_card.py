# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inherit a backend ModelRuntimeConfig into ThunderAgent's public proxy card.

ThunderAgent registers the frontend-facing Chat/Completions surface. The wrapped
worker typically uses ``--endpoint-types none``, so admission uses this proxy
card. Copy the backend runtime contract wholesale (including ``runtime_data``
keys such as ``token_budget``) instead of reconstructing selected fields.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable, Mapping
from typing import Any, Optional

from dynamo.common.token_budget import TOKEN_BUDGET_RUNTIME_KEY

logger = logging.getLogger(__name__)

BACKEND_CARD_WAIT_SECONDS = 30.0
BACKEND_CARD_POLL_INTERVAL_SECONDS = 0.1

_RUNTIME_DATA_KEY = "runtime_data"
_STRUCTURAL_TAG_SETTERS = {
    "structural_tag_mode": "set_structural_tag_mode",
    "structural_tag_scope": "set_structural_tag_scope",
    "structural_tag_schema": "set_structural_tag_schema",
}


def select_backend_card(cards: Mapping[str, str]) -> Optional[str]:
    """Pick a backend MDC JSON body, preferring one that publishes token_budget."""
    preferred: Optional[str] = None
    for card_json in cards.values():
        mapping = _runtime_config_mapping(card_json)
        if mapping is None:
            continue
        runtime_data = mapping.get(_RUNTIME_DATA_KEY)
        if isinstance(runtime_data, dict) and TOKEN_BUDGET_RUNTIME_KEY in runtime_data:
            return card_json
        if preferred is None:
            preferred = card_json
    return preferred


async def wait_for_backend_card(
    get_cards: Callable[[], Mapping[str, str]],
    *,
    timeout_seconds: float = BACKEND_CARD_WAIT_SECONDS,
    poll_interval_seconds: float = BACKEND_CARD_POLL_INTERVAL_SECONDS,
) -> Optional[str]:
    """Poll discovery until a backend card is visible, or *timeout_seconds* elapses."""
    deadline = time.monotonic() + timeout_seconds
    while True:
        selected = select_backend_card(get_cards())
        if selected is not None:
            return selected
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        await asyncio.sleep(min(poll_interval_seconds, remaining))


def runtime_config_from_card_json(
    card_json: str,
    runtime_config_cls: type[Any],
) -> Any:
    """Build a ModelRuntimeConfig that inherits the backend card's runtime contract."""
    cfg = runtime_config_cls()
    mapping = _runtime_config_mapping(card_json)
    if mapping is None:
        raise ValueError("backend model card has no object runtime_config")
    apply_runtime_config_mapping(cfg, mapping)
    return cfg


def kv_cache_block_size_from_card_json(card_json: str) -> Optional[int]:
    """Return the backend card's KV block size when it is a positive int."""
    body = _card_object(card_json)
    if body is None:
        return None
    block_size = body.get("kv_cache_block_size")
    if (
        isinstance(block_size, bool)
        or not isinstance(block_size, int)
        or block_size <= 0
    ):
        return None
    return block_size


def apply_runtime_config_mapping(cfg: Any, mapping: Mapping[str, Any]) -> None:
    """Copy every backend runtime_config field onto *cfg*, including runtime_data."""
    runtime_data = mapping.get(_RUNTIME_DATA_KEY)
    if isinstance(runtime_data, dict):
        _copy_runtime_data(cfg, runtime_data)

    for json_key, setter_name in _STRUCTURAL_TAG_SETTERS.items():
        value = mapping.get(json_key)
        setter = getattr(cfg, setter_name, None)
        if isinstance(value, str) and callable(setter):
            setter(value)

    disagg = mapping.get("disaggregated_endpoint")
    set_disagg = getattr(cfg, "set_disaggregated_endpoint", None)
    if isinstance(disagg, dict) and callable(set_disagg):
        set_disagg(
            bootstrap_host=disagg.get("bootstrap_host"),
            bootstrap_port=disagg.get("bootstrap_port"),
        )

    for key, value in mapping.items():
        if (
            key in {_RUNTIME_DATA_KEY, "disaggregated_endpoint"}
            or key in _STRUCTURAL_TAG_SETTERS
        ):
            continue
        if value is None:
            continue
        assigned = value
        if key == "taints" and isinstance(value, (list, tuple, set)):
            assigned = set(value)
        try:
            setattr(cfg, key, assigned)
        except Exception as exc:
            logger.warning(
                "Skipping backend runtime_config.%s while inheriting proxy card: %s",
                key,
                exc,
            )


def _copy_runtime_data(cfg: Any, runtime_data: Mapping[str, Any]) -> None:
    setter = getattr(cfg, "set_engine_specific", None)
    if not callable(setter):
        return
    for key, value in runtime_data.items():
        setter(key, json.dumps(value))


def _card_object(card_json: str) -> Optional[dict[str, Any]]:
    try:
        body = json.loads(card_json)
    except json.JSONDecodeError:
        return None
    return body if isinstance(body, dict) else None


def _runtime_config_mapping(card_json: str) -> Optional[dict[str, Any]]:
    body = _card_object(card_json)
    if body is None:
        return None
    mapping = body.get("runtime_config") or {}
    return mapping if isinstance(mapping, dict) else None
