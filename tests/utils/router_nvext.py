# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared assertions for router metadata returned in OpenAI `nvext` responses."""

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class RouterNvextExpectation:
    worker_id: bool = False
    kv_block_timing: bool = False


def router_nvext_extra_body(fields: Sequence[str]) -> dict[str, Any]:
    return {"nvext": {"extra_fields": list(dict.fromkeys(fields))}}


def _context_prefix(context: str) -> str:
    return f"{context}: " if context else ""


def response_nvext(data: Mapping[str, Any]) -> Mapping[str, Any]:
    nvext = data.get("nvext") or {}
    assert isinstance(nvext, Mapping), f"Expected nvext object, got {nvext!r}"
    return nvext


def require_router_worker_id(
    data: Mapping[str, Any],
    *,
    context: str = "",
) -> Mapping[str, Any]:
    nvext = response_nvext(data)
    worker_id = nvext.get("worker_id")
    assert isinstance(worker_id, Mapping), (
        f"{_context_prefix(context)}Expected nvext.worker_id object, "
        f"got nvext={dict(nvext)!r}"
    )

    selected_ids = [
        worker_id.get("prefill_worker_id"),
        worker_id.get("decode_worker_id"),
    ]
    assert any(isinstance(selected_id, int) for selected_id in selected_ids), (
        f"{_context_prefix(context)}Expected prefill_worker_id or decode_worker_id "
        f"in nvext.worker_id, got {dict(worker_id)!r}"
    )
    return worker_id


def require_router_kv_block_timing(
    data: Mapping[str, Any],
    *,
    context: str = "",
    recent_logs: str | None = None,
    log_label: str = "router",
) -> tuple[int, int]:
    nvext = response_nvext(data)
    timing = nvext.get("timing") or {}
    assert isinstance(timing, Mapping), (
        f"{_context_prefix(context)}Expected nvext.timing object, "
        f"got nvext={dict(nvext)!r}"
    )

    overlap = timing.get("kv_overlap_blocks")
    total = timing.get("kv_total_blocks")
    if overlap is None or total is None:
        message = (
            f"{_context_prefix(context)}Expected router KV block timing in response "
            f"nvext, got nvext={dict(nvext)!r}."
        )
        if recent_logs is not None:
            message += f"\nRecent {log_label} logs:\n{recent_logs[-4000:]}"
        raise AssertionError(message)

    return int(float(overlap) + 0.5), int(total)


def validate_router_nvext(
    data: Mapping[str, Any],
    expectation: RouterNvextExpectation | None,
    *,
    context: str = "",
) -> None:
    if expectation is None:
        return
    if expectation.worker_id:
        require_router_worker_id(data, context=context)
    if expectation.kv_block_timing:
        require_router_kv_block_timing(data, context=context)
