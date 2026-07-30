# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``dynamo.backend`` — the public SDK for building Dynamo backends.

The curated, semver-committed surface for backend authors, both out-of-tree
(e.g. the ``echo_backend`` example) and in-tree (TokenSpeed, the sample
engines, the sidecar engine glue). Import the engine ABCs, the config /
registration types, the request/response contracts, the ``Worker``, the
``run`` entry point, the runtime ``Context``, and ``DisaggregationMode`` from
the package root.

Stability tiers
---------------
* **Committed surface** — everything in :data:`__all__`. Treated as a
  semver-stable contract; changes go through the deprecation policy. It is
  backed by the private ``_engine`` / ``_worker`` / ``_run`` submodules — import
  from the package root, never from those.
* **Advanced tier** — the helper submodules (:mod:`~dynamo.backend.telemetry`,
  :mod:`~dynamo.backend.disagg`, :mod:`~dynamo.backend.publisher`,
  :mod:`~dynamo.backend.health_check`, :mod:`~dynamo.backend.multimodal`,
  :mod:`~dynamo.backend.logprobs`, :mod:`~dynamo.backend.metrics`,
  :mod:`~dynamo.backend.dp_rank`). Available for richer backends but not yet
  frozen.

Logits-processor serialization internals and the Rust disaggregation-mode
converter live in ``_engine`` / ``_worker`` — framework plumbing, not part of
the author contract.
"""

from dynamo._core import Context
from dynamo.common.constants import DisaggregationMode

from . import disagg as disagg
from . import dp_rank as dp_rank
from . import health_check as health_check
from . import logprobs as logprobs
from . import metrics as metrics
from . import multimodal as multimodal
from . import publisher as publisher
from . import telemetry as telemetry
from ._args import add_worker_args, build_worker_config
from ._engine import (
    BaseEngine,
    DiffusionEngine,
    EngineConfig,
    GenerateChunk,
    GenerateRequest,
    LLMEngine,
    LlmRegistration,
    RawEngine,
    RawRequest,
    RawResponseChunk,
)
from ._run import run
from ._worker import Worker, WorkerConfig

__all__ = [
    "BaseEngine",
    "LLMEngine",
    "RawEngine",
    "DiffusionEngine",
    "EngineConfig",
    "LlmRegistration",
    "GenerateRequest",
    "GenerateChunk",
    "RawRequest",
    "RawResponseChunk",
    "Worker",
    "WorkerConfig",
    "add_worker_args",
    "build_worker_config",
    "run",
    "Context",
    "DisaggregationMode",
]
