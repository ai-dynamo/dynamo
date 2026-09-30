# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prometheus metrics specific to the SGLang embedding worker.

``prometheus_client`` imports are deferred behind helper functions because
this module is loaded via ``dynamo.sglang.request_handlers`` before
``sgl.Engine(...)`` runs its multiprocess-aware Prometheus setup; pulling
``prometheus_client`` in at module load time interferes with that setup.

Histograms are also bound to a caller-provided ``CollectorRegistry`` (NOT
the default global ``REGISTRY``) because the Dynamo SGLang worker exposes
metrics on ``/metrics`` only when the registry has been attached via
``dynamo.common.utils.prometheus.register_engine_metrics_callback``. The
init function below should be called from ``init_embedding.py`` after
engine setup, paired with that callback registration.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from prometheus_client import CollectorRegistry, Histogram

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
_INPUT_TOKENS_BUCKETS = (1, 4, 16, 64, 128, 256, 512, 1024, 2048, 4096, 8192)


_EMBEDDING_BATCH_SIZE: Optional["Histogram"] = None
_EMBEDDING_INPUT_TOKENS: Optional["Histogram"] = None


def init_embedding_metrics(registry: "CollectorRegistry") -> None:
    """Create the embedding histograms against the provided registry.

    Must be called AFTER engine init and ONCE per process. Subsequent
    calls leave the existing histograms alone — this lets tests reuse
    the function without colliding with prior production registration.
    The caller is responsible for wiring ``registry`` to the worker's
    ``/metrics`` endpoint via
    ``dynamo.common.utils.prometheus.register_engine_metrics_callback``.
    """
    from prometheus_client import Histogram

    global _EMBEDDING_BATCH_SIZE, _EMBEDDING_INPUT_TOKENS
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
    if _EMBEDDING_INPUT_TOKENS is None:
        _EMBEDDING_INPUT_TOKENS = Histogram(
            "dynamo_embedding_input_tokens",
            "Total prompt tokens summed across all inputs in a /v1/embeddings "
            "request. One sample per request.",
            labelnames=("model",),
            buckets=_INPUT_TOKENS_BUCKETS,
            registry=registry,
        )


def observe_embedding_batch_size(model: str, batch_size: int) -> None:
    """Record one observation of the request's batch size.

    No-op when ``init_embedding_metrics`` has not been called — keeps
    handler call sites cheap and crash-free in test/CI configurations
    that don't wire metrics.
    """
    if _EMBEDDING_BATCH_SIZE is None:
        return
    if batch_size < 0:
        logger.warning(
            "Skipping batch_size observation with negative value %d", batch_size
        )
        return
    _EMBEDDING_BATCH_SIZE.labels(model=model).observe(batch_size)


def observe_embedding_input_tokens(model: str, input_tokens: int) -> None:
    """Record one observation of the request's total input tokens.

    No-op when ``init_embedding_metrics`` has not been called.
    """
    if _EMBEDDING_INPUT_TOKENS is None:
        return
    if input_tokens < 0:
        logger.warning(
            "Skipping input_tokens observation with negative value %d", input_tokens
        )
        return
    _EMBEDDING_INPUT_TOKENS.labels(model=model).observe(input_tokens)


def reset_metrics_for_testing() -> None:
    """Clear cached Histogram singletons so tests can register against a fresh registry."""
    global _EMBEDDING_BATCH_SIZE, _EMBEDDING_INPUT_TOKENS
    _EMBEDDING_BATCH_SIZE = None
    _EMBEDDING_INPUT_TOKENS = None
