# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integrity tests for the ``dynamo.backend`` public facade.

Pin the committed surface (``__all__``) and assert every root-level re-export is
the same object as its private-submodule source, so importing from the package
root never diverges from the implementation. Also confirm the advanced-tier
helper submodules are importable under the package namespace.
"""

from __future__ import annotations

import importlib

import pytest

import dynamo.backend as facade
from dynamo import _core
from dynamo.backend import _args, _engine, _run, _worker
from dynamo.common import constants

pytestmark = [
    pytest.mark.unit,
    pytest.mark.unified,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]

# A literal (not derived from ``facade.__all__``) so an accidental change to the
# committed surface fails loudly.
COMMITTED_SURFACE = {
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
}

FROM_ENGINE = COMMITTED_SURFACE - {
    "Worker",
    "WorkerConfig",
    "add_worker_args",
    "build_worker_config",
    "run",
    "Context",
    "DisaggregationMode",
}

HELPER_SUBMODULES = (
    "telemetry",
    "disagg",
    "publisher",
    "health_check",
    "multimodal",
    "logprobs",
    "metrics",
    "dp_rank",
)


def test_all_matches_committed_surface():
    assert set(facade.__all__) == COMMITTED_SURFACE


def test_committed_surface_is_identity_reexport():
    for name in FROM_ENGINE:
        assert getattr(facade, name) is getattr(_engine, name), name
    assert facade.Worker is _worker.Worker
    assert facade.WorkerConfig is _worker.WorkerConfig
    assert facade.add_worker_args is _args.add_worker_args
    assert facade.build_worker_config is _args.build_worker_config
    assert facade.run is _run.run
    assert facade.Context is _core.Context
    assert facade.DisaggregationMode is constants.DisaggregationMode


def test_helper_submodules_importable():
    for name in HELPER_SUBMODULES:
        mod = importlib.import_module(f"dynamo.backend.{name}")
        assert getattr(facade, name) is mod, name
