# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared ModelDeploymentCard (MDC) plumbing for planner connectors.

Connectors that want to populate ``WorkerInfo`` from MDCs feed
:class:`MdcEntry` records into :func:`worker_info_from_mdc`.  The transform
is pure: it has no K8s or discovery dependency.  Each connector supplies
its own :class:`MdcSource` implementation that fetches entries from
whatever backing store it uses (Kubernetes CRDs, the dynamo discovery
watch, etc.).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

from dynamo.planner.config.defaults import SubComponentType
from dynamo.planner.monitoring.worker_info import (
    WorkerInfo,
    build_worker_info_from_defaults,
)

logger = logging.getLogger(__name__)

# ModelType::Prefill bit in ModelDeploymentCard::model_type (bitflags: 1 << 4).
_MODEL_TYPE_PREFILL_BIT = 0x10


@dataclass
class MdcEntry:
    """Normalized MDC record consumed by :func:`worker_info_from_mdc`.

    ``component`` / ``endpoint`` come from the CRD wrapper in K8s mode and
    are typically ``None`` for discovery-sourced entries (where they fall
    back to backend defaults).
    """

    card_json: dict = field(default_factory=dict)
    component: Optional[str] = None
    endpoint: Optional[str] = None
    instance_id: Optional[str] = None


class MdcSource(Protocol):
    """Source of :class:`MdcEntry` records scoped to a sub-component type."""

    def get_entries(self, sub_component_type: SubComponentType) -> list[MdcEntry]:
        ...


def is_model_card(wrapper: dict) -> bool:
    """Filter that excludes LoRA-adapter cards from a CRD wrapper dict.

    K8s CRDs store both Model and LoRA cards together; discovery-sourced
    entries are already scoped to the ``Model`` variant, so this filter
    only matters for the CRD path.
    """
    return wrapper.get("type") == "Model"


def is_prefill_card(card_json: dict) -> bool:
    """Whether a card_json belongs to a prefill worker.

    ``model_type`` can be serialized three ways depending on the producer:
    an integer bitflag, a serde-bitflags dict with a ``bits`` key, or a
    human-readable string (e.g. ``"Prefill"`` / ``"Chat|Completions"``).
    """
    model_type: Any = card_json.get("model_type", 0)
    if isinstance(model_type, str):
        return "prefill" in model_type.lower()
    if isinstance(model_type, dict):
        model_type = model_type.get("bits", 0)
    try:
        return bool(int(model_type) & _MODEL_TYPE_PREFILL_BIT)
    except (TypeError, ValueError):
        return False


def select_entry(
    entries: list[MdcEntry],
    sub_component_type: SubComponentType,
    expected_component: Optional[str] = None,
) -> Optional[MdcEntry]:
    """Pick the first entry matching ``sub_component_type`` and (if given)
    ``expected_component``.  Used to scope past LoRA-adapter cards and
    stray entries from sibling deployments.
    """
    want_prefill = sub_component_type == SubComponentType.PREFILL
    for entry in entries:
        if is_prefill_card(entry.card_json) != want_prefill:
            continue
        if (
            entry.component
            and expected_component
            and entry.component != expected_component
        ):
            continue
        return entry
    return None


def worker_info_from_mdc(
    entry: MdcEntry,
    sub_component_type: SubComponentType,
    backend: str,
    model_name_fallback: Optional[Callable[[], Optional[str]]] = None,
    k8s_name_override: Optional[str] = None,
) -> WorkerInfo:
    """Build a :class:`WorkerInfo` from a single :class:`MdcEntry`.

    Pure function.  Connector-specific enrichment is injected via
    ``model_name_fallback`` (called only when the card lacks a
    ``display_name``) and ``k8s_name_override``.
    """
    defaults = build_worker_info_from_defaults(backend, sub_component_type)

    card = entry.card_json or {}
    runtime_cfg = card.get("runtime_config")
    if not isinstance(runtime_cfg, dict):
        runtime_cfg = {}

    component_name = entry.component or defaults.component_name
    endpoint = entry.endpoint or defaults.endpoint

    model_name: Optional[str] = card.get("display_name")
    if not model_name and model_name_fallback is not None:
        try:
            model_name = model_name_fallback()
        except Exception as e:
            logger.debug(f"Model name fallback raised: {e}")
            model_name = None

    k8s_name = k8s_name_override if k8s_name_override is not None else defaults.k8s_name

    return WorkerInfo(
        k8s_name=k8s_name,
        component_name=component_name,
        endpoint=endpoint,
        model_name=model_name,
        total_kv_blocks=runtime_cfg.get("total_kv_blocks"),
        kv_cache_block_size=card.get("kv_cache_block_size"),
        max_num_seqs=runtime_cfg.get("max_num_seqs"),
        max_num_batched_tokens=runtime_cfg.get("max_num_batched_tokens"),
        context_length=card.get("context_length"),
    )
