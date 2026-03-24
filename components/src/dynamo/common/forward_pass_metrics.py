# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
ForwardPassMetrics schema for per-iteration scheduler telemetry.

Uses msgspec.Struct for zero-copy serialization (same approach as KV cache events).
We do not use prometheus for forward pass metrics because:
    1. Metric scrapper for pull based prometheus metrics is async with engine.
       Metrics can be easily lost/repeated.
    2. Push based prometheus uses HTTP and might not scale as well as ZMQ.
    3. Existing KV event infra can be reused for forward pass metrics.

Data flow (two-layer relay, same architecture as KV events)::

    EngineCore child process:
        InstrumentedScheduler -> _FpmPublisherThread -> ZMQ PUB (localhost)

    Dynamo parent process:
        FpmEventRelay (ZMQ SUB) -> EventPublisher -> Event Plane (NATS/ZMQ)

    Consumer (planner, etc.):
        FpmEventSubscriber (auto-discovered) -> decode() -> ForwardPassMetrics

The raw ZMQ hop is needed because the scheduler runs in a forked child
process without access to the Dynamo runtime.  The FpmEventRelay bridge
in the parent process handles event plane transport and discovery
registration automatically.

See ``dynamo.common.recv_forward_pass_metrics`` for a standalone
consumer example.

TODO: add metrics for TrtLLM/SGLang
TODO: planner consuming these metrics instead of frontend/router metrics
"""

from __future__ import annotations

import logging

import msgspec

logger = logging.getLogger(__name__)

FPM_VERSION: int = 1


class WelfordAccumulator:
    """Welford's online algorithm for count / sum / population-variance.

    Numerically stable single-pass computation -- avoids catastrophic
    cancellation that sum-of-squares can suffer with large values.

    Usage::

        acc = WelfordAccumulator()
        for v in values:
            acc.add(v)
        print(acc.n, acc.s, acc.variance())
    """

    __slots__ = ("n", "s", "_mean", "_m2")

    def __init__(self) -> None:
        self.n = 0
        self.s = 0
        self._mean = 0.0
        self._m2 = 0.0

    def add(self, v: int) -> None:
        self.n += 1
        self.s += v
        delta = v - self._mean
        self._mean += delta / self.n
        delta2 = v - self._mean
        self._m2 += delta * delta2

    def variance(self) -> float:
        if self.n == 0:
            return 0.0
        return self._m2 / self.n


class ScheduledRequestMetrics(
    msgspec.Struct,
    frozen=True,  # type: ignore[call-arg]
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
    frozen=True,  # type: ignore[call-arg]
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
    frozen=True,  # type: ignore[call-arg]
    gc=False,
):
    """Per-iteration metrics emitted by InstrumentedScheduler.

    One message is emitted per scheduler iteration (one per forward pass).
    An idle heartbeat (all zeros, wall_time=0) is emitted once when the
    engine transitions from active to idle.
    """

    # Schema version. Consumers must check this before interpreting
    # the remaining fields. Bump when the schema changes incompatibly.
    version: int = FPM_VERSION

    # Unique worker identifier (Dynamo runtime connection_id).
    worker_id: str = ""

    # Data parallel rank. Each DP rank has its own scheduler and ZMQ port.
    dp_rank: int = 0

    # Monotonically increasing sequence number per (worker_id, dp_rank).
    # Set by _FpmPublisherThread before encoding; 0 for messages that
    # have not been stamped (e.g. unit-test fixtures).
    counter_id: int = 0

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


class UnsupportedFpmVersionError(Exception):
    """Raised when a ForwardPassMetrics message has an unrecognised version."""


def decode(data: bytes) -> ForwardPassMetrics | None:
    """Decode a ForwardPassMetrics message, returning None for unknown versions.

    Returns None (and logs a warning) if the message cannot be decoded or
    carries a version this code does not understand, so callers can simply
    skip unsupported messages without crashing.
    """
    try:
        metrics = _decoder.decode(data)
    except Exception:
        logger.warning("Failed to decode ForwardPassMetrics message, skipping")
        return None
    if metrics.version != FPM_VERSION:
        logger.warning(
            "Unsupported ForwardPassMetrics version %d (expected %d), skipping",
            metrics.version,
            FPM_VERSION,
        )
        return None
    return metrics
