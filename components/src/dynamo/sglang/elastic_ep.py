# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Forward Elastic EP controls without changing native replay or result semantics."""

from typing import Any


async def scale_external_elastic_ep(tokenizer_manager: Any, body: object) -> dict:
    """Validate the request, then preserve the engine's correlated result.

    Invocation errors propagate: they do not prove the collective was rejected.
    """

    def err(message: str) -> dict:
        return {"status": "error", "message": message}

    if body is None:
        body = {}
    if not isinstance(body, dict):
        return err("request body must be a JSON object")

    new_ep_size = body.get("new_ep_size")
    if new_ep_size is None:
        return err("Missing required field: new_ep_size")
    # bool is an int subclass — reject it so True/False can't pose as a size.
    if isinstance(new_ep_size, bool) or not isinstance(new_ep_size, int):
        return err(f"new_ep_size must be an integer, got: {new_ep_size!r}")
    if new_ep_size <= 0:
        return err("new_ep_size must be a positive integer")

    # Pass native lifecycle correlation through unchanged; the engine owns replay and conflict handling.
    correlation: dict[str, Any] = {}
    for field in ("operation_id", "expected_instance_id", "expected_topology_token"):
        value = body.get(field)
        if value is not None:
            if not isinstance(value, str) or not value:
                return err(f"{field} must be a non-empty string")
            correlation[field] = value
    allocations = body.get("expected_joining_allocation_ids")
    if allocations is not None:
        if not isinstance(allocations, list) or not all(
            isinstance(value, str) and value for value in allocations
        ):
            return err(
                "expected_joining_allocation_ids must be a list of non-empty strings"
            )
        correlation["expected_joining_allocation_ids"] = allocations

    from sglang.srt.managers.io_struct import ScaleElasticEPReqInput

    result = await tokenizer_manager.scale_elastic_ep(
        ScaleElasticEPReqInput(new_ep_size=new_ep_size, **correlation)
    )

    message = result.message
    if not message:
        message = (
            f"Scaled to ep_size={new_ep_size}"
            if result.success
            else "scale_elastic_ep failed"
        )
    response = {
        "status": "ok" if result.success else "error",
        "message": message,
        "old_ep_size": result.old_ep_size,
        "new_ep_size": result.new_ep_size,
    }
    if not result.success:
        response["pending_ep_size"] = result.pending_ep_size
    if correlation:
        response.update(
            operation_id=result.operation_id,
            instance_id=result.instance_id,
            scale_phase=result.scale_phase,
            terminal=result.terminal,
            effective_ep_size=result.effective_ep_size,
            pending_ep_size=result.pending_ep_size,
            conflict=result.conflict,
        )
    return response
