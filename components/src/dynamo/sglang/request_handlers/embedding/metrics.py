# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prometheus metrics specific to the SGLang embedding worker.

The existing SGLang publisher emits chat-shaped metrics (KV gauges,
prefill/decode counters) which are always zero on a pooling engine.
The histograms here capture the signals that actually move on an
embedding worker: how many inputs per request, and how many input
tokens those inputs amount to.

A third histogram (per-request embedding latency, with a `worker_id`
label) is more naturally implemented on the Rust frontend side
because that's where per-worker timing is observable end-to-end;
that piece is tracked separately. The two histograms below cover
the worker-internal view.

Metric names follow Dynamo's existing ``dynamo_<scope>_<unit>``
convention. They are intentionally NOT prefixed with ``sglang:``
because they describe an OpenAI-spec workload shape, not an
SGLang-engine-internal signal — once the vLLM embedding worker
ships (see PR #9713), it will observe the same names from its own
handler.
"""

from __future__ import annotations

import logging
from typing import Optional

from prometheus_client import REGISTRY, CollectorRegistry, Histogram

logger = logging.getLogger(__name__)

# Bucket choices:
#
# batch_size: clients typically send 1 input (chat-style embedding lookup),
# OpenAI's hosted limit is 2048; powers-of-two up to that covers both
# common cases and the rare big batches.
_BATCH_SIZE_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)

# input_tokens (summed across all inputs in a request): typical embedding
# ISLs are 60-200 (sentence-level); OpenAI's per-input limit is 8192
# tokens. Buckets span 1..8K with denser coverage in the common range.
_INPUT_TOKENS_BUCKETS = (
    1,
    4,
    16,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
)


_EMBEDDING_BATCH_SIZE: Optional[Histogram] = None
_EMBEDDING_INPUT_TOKENS: Optional[Histogram] = None


def _get_or_create_batch_size_histogram(
    registry: CollectorRegistry,
) -> Histogram:
    global _EMBEDDING_BATCH_SIZE
    if _EMBEDDING_BATCH_SIZE is None:
        _EMBEDDING_BATCH_SIZE = Histogram(
            "dynamo_embedding_batch_size",
            "Number of inputs per /v1/embeddings request, observed when the "
            "worker successfully transforms the engine output. One sample per "
            "request, not per input.",
            labelnames=("model",),
            buckets=_BATCH_SIZE_BUCKETS,
            registry=registry,
        )
    return _EMBEDDING_BATCH_SIZE


def _get_or_create_input_tokens_histogram(
    registry: CollectorRegistry,
) -> Histogram:
    global _EMBEDDING_INPUT_TOKENS
    if _EMBEDDING_INPUT_TOKENS is None:
        _EMBEDDING_INPUT_TOKENS = Histogram(
            "dynamo_embedding_input_tokens",
            "Total prompt tokens summed across all inputs in a /v1/embeddings "
            "request. One sample per request.",
            labelnames=("model",),
            buckets=_INPUT_TOKENS_BUCKETS,
            registry=registry,
        )
    return _EMBEDDING_INPUT_TOKENS


def observe_embedding_batch_size(
    model: str,
    batch_size: int,
    *,
    registry: CollectorRegistry = REGISTRY,
) -> None:
    """Record one observation of the request's batch size.

    ``registry`` is overridable for tests; production code uses the
    default Prometheus global registry.
    """
    if batch_size < 0:
        logger.warning(
            "Skipping batch_size observation with negative value %d", batch_size
        )
        return
    _get_or_create_batch_size_histogram(registry).labels(model=model).observe(
        batch_size
    )


def observe_embedding_input_tokens(
    model: str,
    input_tokens: int,
    *,
    registry: CollectorRegistry = REGISTRY,
) -> None:
    """Record one observation of the request's total input tokens.

    ``registry`` is overridable for tests; production code uses the
    default Prometheus global registry.
    """
    if input_tokens < 0:
        logger.warning(
            "Skipping input_tokens observation with negative value %d", input_tokens
        )
        return
    _get_or_create_input_tokens_histogram(registry).labels(model=model).observe(
        input_tokens
    )


def reset_metrics_for_testing() -> None:
    """Clear cached Histogram singletons so tests can register against a fresh registry."""
    global _EMBEDDING_BATCH_SIZE, _EMBEDDING_INPUT_TOKENS
    _EMBEDDING_BATCH_SIZE = None
    _EMBEDDING_INPUT_TOKENS = None
