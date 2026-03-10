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
    """Metrics for requests scheduled in this iteration"""

    # Number of prefill requests (new requests + chunked prefill continuations).
    num_prefill_requests: int = 0

    # Total tokens being freshly computed for prefill requests in this
    # iteration. Does NOT include prefix-cached or previously-chunked tokens
    # (those are in sum_prefill_kv_tokens). For chunked prefill, this is the
    # chunk size being computed this step.
    sum_prefill_tokens: int = 0

    # Population variance of total prompt lengths (not chunk sizes) across
    # prefill requests. A request with a 10k-token prompt counts as 10k even
    # if only a 2k chunk is computed this iteration.
    var_prefill_length: float = 0.0

    # Total KV cache tokens that must be read (not computed) for prefill
    # requests. Includes prefix cache hits for new requests and previously
    # computed chunks for chunked prefill continuations.
    sum_prefill_kv_tokens: int = 0

    # Number of decode requests (generating output tokens).
    num_decode_requests: int = 0

    # Total KV context length across all decode requests (prompt + generated
    # tokens so far). Reflects the memory pressure from decoding.
    sum_decode_kv_tokens: int = 0

    # Population variance of KV context lengths across decode requests.
    # High variance means a mix of short and long sequences decoding together.
    var_decode_kv_tokens: float = 0.0


class QueuedRequestMetrics(
    msgspec.Struct,
    gc=False,
):
    """Metrics for requests waiting in the queue (not scheduled this iteration).

    All token counts here are raw totals -- prefix cache effects are unknown
    until a request is actually scheduled.
    """

    # Number of queued prefill requests (status=WAITING).
    num_prefill_requests: int = 0

    # Total prompt token count of queued prefill requests.
    sum_prefill_tokens: int = 0

    # Population variance of prompt lengths for queued prefill requests.
    var_prefill_length: float = 0.0

    # Number of queued decode requests (preempted -- were decoding but got
    # evicted back to the waiting queue due to memory pressure).
    num_decode_requests: int = 0

    # Total KV context length of queued decode (preempted) requests.
    sum_decode_kv_tokens: int = 0

    # Population variance of KV context lengths for queued decode requests.
    var_decode_kv_tokens: float = 0.0


class ForwardPassMetrics(
    msgspec.Struct,
    gc=False,
):
    """Per-iteration metrics emitted by InstrumentedScheduler.

    One message is emitted per scheduler iteration (one per forward pass).
    An idle heartbeat (all zeros, wall_time=0) is emitted once when the
    engine transitions from active to idle.
    """

    # Unique worker identifier (Dynamo runtime connection_id).
    worker_id: str = ""

    # Data parallel rank. Each DP rank has its own scheduler and ZMQ port.
    dp_rank: int = 0

    # Wall-clock time of this iteration: from schedule() to update_from_output().
    # Covers scheduling + model forward pass + output processing.
    # 0.0 for idle heartbeat messages.
    wall_time: float = 0.0

    # Requests that were scheduled and executed in this iteration.
    scheduled_requests: ScheduledRequestMetrics = ScheduledRequestMetrics()

    # Requests that exist in the waiting queue but were not scheduled.
    queued_requests: QueuedRequestMetrics = QueuedRequestMetrics()


_encoder = msgspec.msgpack.Encoder()
_decoder = msgspec.msgpack.Decoder(ForwardPassMetrics)


def encode(metrics: ForwardPassMetrics) -> bytes:
    return _encoder.encode(metrics)


def decode(data: bytes) -> ForwardPassMetrics:
    return _decoder.decode(data)
