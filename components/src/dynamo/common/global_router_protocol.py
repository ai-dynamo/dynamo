# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared request/control fields for frontend <-> global-router coordination."""

from typing import Any, Mapping, Optional

GLOBAL_ROUTER_CONTROL_FIELD = "global_router_control"
GLOBAL_ROUTER_RETRY_ATTEMPT_KEY = "global_router_retry_attempt"

GLOBAL_ROUTER_ACTION_RETRY = "retry"
GLOBAL_ROUTER_ACTION_EXHAUSTED = "exhausted"


def get_global_router_retry_attempt(request: Mapping[str, Any]) -> Optional[int]:
    routing = request.get("routing") or {}
    if not isinstance(routing, Mapping):
        return None
    if GLOBAL_ROUTER_RETRY_ATTEMPT_KEY not in routing:
        return None

    value = routing[GLOBAL_ROUTER_RETRY_ATTEMPT_KEY]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{GLOBAL_ROUTER_RETRY_ATTEMPT_KEY} must be an integer")
    retry_attempt = value
    if retry_attempt < 0:
        raise ValueError(f"{GLOBAL_ROUTER_RETRY_ATTEMPT_KEY} must be >= 0")
    return retry_attempt


def make_global_router_retry_control(
    *,
    request_type: str,
    retry_attempt: int,
    next_retry_attempt: int,
    failed_pool: int,
    failed_namespace: str,
    next_pool: int,
    next_namespace: str,
    error: str,
) -> dict[str, Any]:
    return {
        GLOBAL_ROUTER_CONTROL_FIELD: {
            "action": GLOBAL_ROUTER_ACTION_RETRY,
            "request_type": request_type,
            "retry_attempt": retry_attempt,
            "next_retry_attempt": next_retry_attempt,
            "failed_pool": failed_pool,
            "failed_namespace": failed_namespace,
            "next_pool": next_pool,
            "next_namespace": next_namespace,
            "error": error,
        }
    }


def make_global_router_retry_exhausted_control(
    *,
    request_type: str,
    retry_attempt: int,
    error: str,
    failed_pool: Optional[int] = None,
    failed_namespace: Optional[str] = None,
) -> dict[str, Any]:
    return {
        GLOBAL_ROUTER_CONTROL_FIELD: {
            "action": GLOBAL_ROUTER_ACTION_EXHAUSTED,
            "request_type": request_type,
            "retry_attempt": retry_attempt,
            "failed_pool": failed_pool,
            "failed_namespace": failed_namespace,
            "error": error,
        }
    }


def get_global_router_control(data: Any) -> Optional[Mapping[str, Any]]:
    if not isinstance(data, Mapping):
        return None
    control = data.get(GLOBAL_ROUTER_CONTROL_FIELD)
    if not isinstance(control, Mapping):
        return None
    return control
