# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Forward external Elastic EP controls to vLLM's native transaction authority.

This module owns only request validation and serialization. In particular, an
exception is not a definitive rejection, and a native failure is not a reason to
replay, roll back, or restart the engine here. The caller observes native status
to resolve ambiguous outcomes. Correlated controls require vLLM #62082.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import msgspec

try:
    from vllm.distributed.elastic_ep.external_elastic_ep import (
        ExternalElasticEPResizeRequest,
    )
    from vllm.entrypoints.serve.elastic_ep.middleware import (
        get_elastic_ep_rank_retired,
        get_scaling_elastic_ep,
    )
except ImportError:
    # Ordinary workers also run against vLLM versions without external Elastic EP.
    get_elastic_ep_rank_retired = None
    get_scaling_elastic_ep = None
    ExternalElasticEPResizeRequest = None

if TYPE_CHECKING:
    from vllm.v1.engine.async_llm import AsyncLLM


_STATUS_FIELDS = frozenset(
    {
        "instance_id",
        "operation_id",
        "epoch",
        "phase",
        "requested_data_parallel_size",
        "effective_data_parallel_size",
        "topology_known",
        "error",
    }
)


@dataclass(frozen=True)
class _ScaleRequest:
    size: int
    drain_timeout: int
    operation_id: str
    expected_instance_id: str

    @classmethod
    def parse(cls, body: object) -> _ScaleRequest:
        if not isinstance(body, dict):
            raise TypeError("request body must be a JSON object")
        return cls(
            size=_positive_integer(
                body.get("new_data_parallel_size"), "new_data_parallel_size"
            ),
            drain_timeout=_positive_integer(
                body.get("drain_timeout", 120), "drain_timeout"
            ),
            operation_id=_identifier(body, "operation_id"),
            expected_instance_id=_identifier(body, "expected_instance_id"),
        )


def _positive_integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def require_native_admission() -> tuple[Callable[[], bool], Callable[[], bool]]:
    if get_elastic_ep_rank_retired is None or get_scaling_elastic_ep is None:
        raise RuntimeError(
            "Engine Group serving requires vLLM external Elastic EP admission support"
        )
    return get_scaling_elastic_ep, get_elastic_ep_rank_retired


def native_allows_requests() -> bool:
    scaling, retired = require_native_admission()
    return not scaling() and not retired()


def require_native_submission() -> type:
    if ExternalElasticEPResizeRequest is None:
        raise RuntimeError(
            "Engine Group resizing requires vLLM's bounded native submission API"
        )
    return ExternalElasticEPResizeRequest


def _identifier(body: dict, name: str) -> str:
    value = body.get(name)
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError(f"{name} must contain 1–256 nonblank characters")
    return value


def uses_external_elastic_ep(engine: AsyncLLM) -> bool:
    """Select native external ownership without changing the legacy Ray path."""
    parallel = engine.vllm_config.parallel_config
    return parallel.enable_elastic_ep and (
        parallel.data_parallel_external_lb
        or parallel.data_parallel_backend == "mp"
        and parallel.elastic_ep_auto_bootstrap
    )


def _status_payload(status: object) -> dict[str, Any]:
    payload = msgspec.to_builtins(status)
    if not isinstance(payload, dict) or not _STATUS_FIELDS.issubset(payload):
        raise RuntimeError(
            "external Elastic EP requires vLLM's correlated operation-status API "
            "(vllm-project/vllm#62082)"
        )
    # Preserve native phase, identity, operation, and topology certainty exactly.
    # In particular, failed != rejected and completed != verified/admitted.
    payload["is_scaling_elastic_ep"] = payload["phase"] in {
        "preparing",
        "committing",
    }
    return payload


async def scale_external_elastic_ep(engine: AsyncLLM, body: object) -> dict[str, Any]:
    """Forward one absolute, identity-bound request; vLLM owns serialization/replay."""
    request = _ScaleRequest.parse(body)
    status = await engine.scale_elastic_ep(
        request.size,
        drain_timeout=request.drain_timeout,
        operation_id=request.operation_id,
        expected_instance_id=request.expected_instance_id,
    )
    return _status_payload(status)


async def submit_external_elastic_ep(engine: AsyncLLM, body: object) -> dict[str, Any]:
    """Return the native client's receipt; execution remains owned by vLLM."""
    if not isinstance(body, dict):
        raise TypeError("request body must be a JSON object")
    request = _ScaleRequest.parse(body)
    native_type = require_native_submission()
    native_request = native_type(
        new_data_parallel_size=request.size,
        drain_timeout=request.drain_timeout,
        operation_id=request.operation_id,
        expected_instance_id=request.expected_instance_id,
        base_topology_token=_identifier(body, "base_topology_token"),
    )
    receipt = msgspec.to_builtins(await engine.submit_elastic_ep(native_request))
    if not isinstance(receipt, dict):
        raise RuntimeError("native Elastic EP returned an invalid acceptance receipt")
    return receipt


async def observe_external_elastic_ep(engine: AsyncLLM) -> dict[str, Any]:
    """Read native state independently of an in-flight scale request."""
    return _status_payload(await engine.get_external_elastic_ep_status())
