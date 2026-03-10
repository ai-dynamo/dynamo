# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
ForwardPassMetrics schema for per-iteration scheduler telemetry.

Published over ZMQ PUB by InstrumentedScheduler, consumed by the
planner or any ZMQ SUB listener.

Uses msgspec.Struct for zero-copy serialization (same approach as
vLLM's KV cache events).
"""

from __future__ import annotations

import msgspec


class ScheduledRequestMetrics(
    msgspec.Struct,
    gc=False,
):
    num_prefill_requests: int = 0
    sum_prefill_tokens: int = 0
    var_prefill_length: float = 0.0
    sum_prefill_tokens_prefix_cached: int = 0
    var_non_prefix_cached_prefill_length: float = 0.0
    num_decode_requests: int = 0
    sum_decode_kv_tokens: int = 0
    sum_decode_kv_tokens_prefix_cached: int = 0
    var_decode_kv_tokens: float = 0.0
    var_decode_kv_tokens_prefix_cached: float = 0.0


class QueuedRequestMetrics(
    msgspec.Struct,
    gc=False,
):
    num_prefill_requests: int = 0
    sum_prefill_tokens: int = 0
    var_prefill_length: float = 0.0
    num_decode_requests: int = 0
    sum_decode_kv_tokens: int = 0
    var_decode_kv_tokens: float = 0.0


class ForwardPassMetrics(
    msgspec.Struct,
    gc=False,
):
    worker_id: str = ""
    dp_rank: int = 0
    wall_time: float = 0.0
    scheduled_requests: ScheduledRequestMetrics = ScheduledRequestMetrics()
    queued_requests: QueuedRequestMetrics = QueuedRequestMetrics()


_encoder = msgspec.msgpack.Encoder()
_decoder = msgspec.msgpack.Decoder(ForwardPassMetrics)


def encode(metrics: ForwardPassMetrics) -> bytes:
    return _encoder.encode(metrics)


def decode(data: bytes) -> ForwardPassMetrics:
    return _decoder.decode(data)
