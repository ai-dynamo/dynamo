# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire format for the cost-eval request-plane endpoint.

Field names + types mirror the Rust ``CostEvalRequest`` / ``CostEvalResponse``
structs in ``lib/kv-router/src/conditional_prefill.rs``. Keep them in sync —
the Rust side is the source of truth and any schema change must be paired
with an endpoint-name bump (see ``ENDPOINT_NAME``).

The dynamo request plane handles serde marshalling between Rust callers and
the Python endpoint, so we just declare Pydantic models — no encode/decode
helpers needed. This matches the planner's precedent at
``components/src/dynamo/planner/__main__.py``.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

# Endpoint suffix. The router's slow-path RPC targets
# ``{namespace}.cost_eval.{ENDPOINT_NAME}``. Bump the version (e.g.
# ``evaluate_v2``) on any schema-breaking change so older routers don't talk
# to newer services or vice versa.
ENDPOINT_NAME = "evaluate_v1"

# Component name used when the cost-eval service registers its endpoint.
COMPONENT_NAME = "cost_eval"


class CostEvalRequest(BaseModel):
    """Slow-path features sent by the router.

    Mirrors the Rust ``CostEvalRequest`` struct field-for-field.
    """

    request_id: str
    prompt_tokens: int
    agg_kv_hit_rate: float
    disagg_kv_hit_rate: float
    decode_chosen_worker_id: int
    decode_chosen_dp_rank: int
    prefill_chosen_worker_id: Optional[int] = None
    prefill_chosen_dp_rank: Optional[int] = None


class CostEvalResponse(BaseModel):
    """Slow-path verdict returned to the router.

    Mirrors the Rust ``CostEvalResponse`` struct field-for-field.

    ``agg_ttft_ms`` / ``disagg_ttft_ms`` are predicted TTFTs in milliseconds.
    Either may be ``None`` when its regression isn't fitted yet. The two
    ``*_warm`` flags expose warmth explicitly so the router can apply its
    conservative-DISAGG fallback without inspecting the ms values.
    """

    agg_ttft_ms: Optional[float] = None
    disagg_ttft_ms: Optional[float] = None
    # Total predicted cost per side: ``ttft_ms + itl_ms * avg_decode_length``.
    # AGG-side ITL is the 2D agg regression queried at the chunked point;
    # DISAGG-side ITL is the *same* 2D regression queried at the pure-decode
    # slice (prefill_tokens=0). They generally differ. ``None`` when any
    # component (TTFT, ITL, or avg_decode_length) couldn't be computed; the
    # router falls back to TTFT-only comparison when total_cost is missing.
    agg_total_cost_ms: Optional[float] = None
    disagg_total_cost_ms: Optional[float] = None
    agg_warm: bool = False
    disagg_warm: bool = False

    @classmethod
    def unavailable(cls) -> "CostEvalResponse":
        """Sentinel returned when regressions aren't ready or the request can
        not be served. The router treats this the same as a transport failure:
        conservative DISAGG."""
        return cls(
            agg_ttft_ms=None,
            disagg_ttft_ms=None,
            agg_total_cost_ms=None,
            disagg_total_cost_ms=None,
            agg_warm=False,
            disagg_warm=False,
        )
