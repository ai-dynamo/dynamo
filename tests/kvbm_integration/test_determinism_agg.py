#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Determinism test for KVBM in aggregated mode.

To make sure KVBM's accuracy, this test suite checks if the model produces
deterministic outputs when same requests are served 1) without KVBM onboarded KV
blocks and 2) with KVBM onboarded KV blocks, when given the same inputs with
fixed seed and temperature=0.

The expected results should be 100% match between the two cases. Compared to
disaggregated mode, aggregated mode has less randomness chances.

These tests are slow by default (~368s and ~601s). For faster runs with
fewer iterations, run the following command (expected to finish in ~58s + ~152s):

    KVBM_MAX_ITERATIONS=2 KVBM_NUM_ITERATIONS=2 KVBM_REQUEST_DELAY=2 \
        pytest tests/kvbm_integration/test_determinism_agg.py -v --tb=short

The harness uses the three-layer fixture decomposition documented in
`tests/kvbm_integration/README.md` (deps / server / eval). Local iteration
can run each layer in a separate shell via `tests/kvbm_integration/scripts/`.
"""

import os
from pathlib import Path
from typing import List

import pytest

from .common import TestDeterminism as BaseTestDeterminism
from .common import check_module_available
from .fixtures import KvbmModelConfig, KvbmServerSpec

HAS_VLLM_BENCH = check_module_available("vllm")


# Models exercised by this test suite.
# CI iterates over all entries; add a new entry to test an additional model.
_MODEL_CONFIGS: List[KvbmModelConfig] = [
    KvbmModelConfig(
        model_id=os.environ.get(
            "KVBM_MODEL_ID", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        ),
        block_size=16,
        attention_backend="FLASH_ATTN",
    ),
    KvbmModelConfig(
        model_id="deepseek-ai/DeepSeek-V2-Lite",
        # TRITON_MLA works on all devices; on H100 set KVBM_MLA_BACKEND=FLASH_ATTN_MLA
        attention_backend=os.environ.get("KVBM_MLA_BACKEND", "TRITON_MLA"),
        # VLLM_BATCH_INVARIANT=1 disables prefix caching for TRITON_MLA in vLLM 0.17.1
        batch_invariant=False,
    ),
]

# KVBM env vars that drive test duration (used to compute timeouts below).
_KVBM_MAX_ITERATIONS = int(os.environ.get("KVBM_MAX_ITERATIONS", "100"))
_KVBM_NUM_ITERATIONS = int(os.environ.get("KVBM_NUM_ITERATIONS", "15"))
_KVBM_REQUEST_DELAY = int(os.environ.get("KVBM_REQUEST_DELAY", "30"))

# Server startup timeout (env-configurable; larger models like DeepSeek-V2-Lite
# may need 600s+).
_SERVER_START_TIMEOUT = int(os.environ.get("KVBM_SERVER_START_TIMEOUT", "600"))

# Compute timeouts from the same env vars that control test duration.
# Each formula adds _SERVER_START_TIMEOUT so the pytest timeout covers both
# the server startup and the actual test body.
#
# test_determinism_agg_with_cache_reset: warmup + 2 phases of KVBM_MAX_ITERATIONS,
# each iteration ~4s (request + overhead), plus ~50s teardown.
_CACHE_RESET_TIMEOUT = _SERVER_START_TIMEOUT + 2 * (_KVBM_MAX_ITERATIONS * 4 + 50)
# test_concurrent_determinism_under_load: dominated by
# (KVBM_NUM_ITERATIONS - 1) * KVBM_REQUEST_DELAY seconds of sleep,
# plus ~150s overhead (benchmark ramp, teardown).
_CONCURRENT_TIMEOUT = _SERVER_START_TIMEOUT + 2 * (
    (_KVBM_NUM_ITERATIONS - 1) * _KVBM_REQUEST_DELAY + 150
)

# Test markers to align with repository conventions.
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.slow,
    pytest.mark.gpu_1,
    pytest.mark.nightly,
]


# Phase 5: v1 and v2 both enumerated. v2 specs are further crossed with
# both onboard modes so every run exercises intra AND inter paths.
_KVBM_VERSIONS_UNDER_TEST = ("v1", "v2")

# v2 onboard modes enumerated on every run. Phase-4 validated structural
# bring-up for both; phase 5 validates determinism numerics for both so any
# mode-specific regression is caught immediately.
_KVBM_V2_ONBOARD_MODES = ("intra", "inter")

# MLA execution is gated for phase 3 (see ACTIVE_PLAN.md). The MLA spec
# stays in _MODEL_CONFIGS so the test surface is preserved; specs whose
# model_config.use_mla is True are pytest-skipped unless KVBM_ENABLE_MLA
# is set. A later phase will flip the gate on.
_KVBM_ENABLE_MLA = os.environ.get("KVBM_ENABLE_MLA", "").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
_MLA_SKIP_REASON = (
    "MLA gated; set KVBM_ENABLE_MLA=1 to enable (see ACTIVE_PLAN.md phase 3)"
)


def _specs(cpu_blocks_env: str, cpu_blocks_default: str) -> List[KvbmServerSpec]:
    cpu_blocks = int(os.environ.get(cpu_blocks_env, cpu_blocks_default))
    gpu_blocks = int(os.environ.get("KVBM_GPU_BLOCKS", "2048"))
    specs: List[KvbmServerSpec] = []
    for version in _KVBM_VERSIONS_UNDER_TEST:
        for cfg in _MODEL_CONFIGS:
            if version == "v2":
                # Cross with both onboard modes — one spec per (model, mode).
                for mode in _KVBM_V2_ONBOARD_MODES:
                    specs.append(
                        KvbmServerSpec(
                            kvbm_version=version,
                            model_config=cfg,
                            cpu_blocks=cpu_blocks,
                            gpu_blocks=gpu_blocks,
                            onboard_mode=mode,
                        )
                    )
            else:
                specs.append(
                    KvbmServerSpec(
                        kvbm_version=version,
                        model_config=cfg,
                        cpu_blocks=cpu_blocks,
                        gpu_blocks=gpu_blocks,
                    )
                )
    return specs


def _params(specs: List[KvbmServerSpec]):
    """Wrap specs as pytest.param objects, applying the MLA skip mark when gated."""
    out = []
    for spec in specs:
        marks = []
        if spec.model_config.use_mla and not _KVBM_ENABLE_MLA:
            marks.append(pytest.mark.skip(reason=_MLA_SKIP_REASON))
        out.append(pytest.param(spec, id=spec.id, marks=marks))
    return out


# Raw spec lists are the contract used by tests/kvbm_integration/scripts/run_server.sh
# to look up a KvbmServerSpec by id without importing pytest internals.
_CACHE_RESET_SPECS = _specs("KVBM_CPU_BLOCKS", "10000")
_CONCURRENT_SPECS = _specs("KVBM_CPU_BLOCKS", "30000")
_CACHE_RESET_PARAMS = _params(_CACHE_RESET_SPECS)
_CONCURRENT_PARAMS = _params(_CONCURRENT_SPECS)


class TestDeterminismAgg(BaseTestDeterminism):
    """Aggregated-mode determinism validation."""

    @pytest.mark.parametrize(
        "kvbm_server_spec",
        _CACHE_RESET_PARAMS,
        indirect=True,
    )
    @pytest.mark.kvbm
    @pytest.mark.timeout(_CACHE_RESET_TIMEOUT)  # ~368s actual on 32-core machine
    def test_determinism_agg_with_cache_reset(self, kvbm_tester, kvbm_server):
        """Run test with warmup, reset cache, run again without warmup.

        Note: `runtime_services` is brought up transitively through `kvbm_deps`
        (see fixtures/deps.py); it is not a direct test parameter so that
        external-attach mode (KVBM_EXTERNAL_BASE_URL) can fully bypass spawn.
        The base method's `runtime_services` arg is unused in its body — we
        pass `None` positionally.
        """
        super().base_test_determinism_with_cache_reset(kvbm_tester, kvbm_server, None)

    @pytest.mark.parametrize(
        "kvbm_server_spec",
        _CONCURRENT_PARAMS,
        indirect=True,
    )
    @pytest.mark.kvbm_concurrency
    @pytest.mark.skipif(
        not HAS_VLLM_BENCH, reason="requires vllm bench (vllm module not found)"
    )
    @pytest.mark.timeout(_CONCURRENT_TIMEOUT)  # ~601s actual on 32-core machine
    def test_concurrent_determinism_under_load(self, kvbm_tester, kvbm_server):
        """Spanish prompt determinism under high concurrency load.

        Reproduces the bug where Spanish responses become English or corrupted.
        """
        spanish_prompt_path = Path(
            os.path.join(os.path.dirname(__file__), "es_prompt.txt")
        ).absolute()
        super().base_test_spanish_prompt_determinism_under_load(
            kvbm_tester, kvbm_server, None, spanish_prompt_path
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
