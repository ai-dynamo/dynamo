# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tiny helpers shared across unified-backend engines for disagg dispatch.

Each engine's `generate()` consults `WorkerConfig.disaggregation_mode` and
branches on it. The functions below capture the two patterns that recur in
multiple engines so we don't reinvent them per backend:

* :func:`enforce_prefill_max_tokens` — clamp `stop_conditions.max_tokens`
  to 1 in PREFILL mode. Prefill workers only need the KV cache populated for
  the prompt; producing more than one token is wasted compute and breaks
  the prefill→decode handoff contract.

* :func:`extract_prefill_result` — pull the optional ``prefill_result`` dict
  out of the request.  Returns ``None`` when the request didn't pass through
  the frontend's prefill router (e.g. aggregated mode).

These are utilities, not abstractions. Backends are free to inline the
behavior if their generate path is shaped differently.
"""

from __future__ import annotations

from typing import Any, Optional

from dynamo.common.backend.engine import GenerateRequest
from dynamo.common.constants import DisaggregationMode


def enforce_prefill_max_tokens(request: GenerateRequest) -> GenerateRequest:
    """In-place clamp ``stop_conditions.max_tokens`` to 1.

    Caller is responsible for only invoking this on prefill workers — the
    helper does not check the disaggregation mode itself.
    """
    stop = request.setdefault("stop_conditions", {})  # type: ignore[typeddict-item]
    stop["max_tokens"] = 1
    return request


def extract_prefill_result(request: GenerateRequest) -> Optional[dict[str, Any]]:
    """Return the request's ``prefill_result`` dict, or ``None`` if absent.

    The frontend's prefill router sets this on decode-bound requests; engines
    use the embedded ``disaggregated_params`` to resume from the KV cache the
    prefill peer already populated.
    """
    return request.get("prefill_result")  # type: ignore[return-value]


def require_prefill_result(
    request: GenerateRequest, mode: DisaggregationMode
) -> dict[str, Any]:
    """Like :func:`extract_prefill_result` but raises when the request is
    missing the prefill handoff payload — the canonical decode-mode
    pre-condition. Pass the mode in so the error message can name the role.
    """
    prefill_result = extract_prefill_result(request)
    if prefill_result is None:
        raise ValueError(
            f"{mode.value} worker received request with no prefill_result; "
            "expected the frontend's prefill router to forward "
            "disaggregated_params from a prefill peer"
        )
    return prefill_result
