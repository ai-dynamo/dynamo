# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the direct Sweeper-to-DGD renderer.

The candidate fixtures below are copied verbatim from a real
aisimulate.sweeper.Sweeper.run() invocation (scalar and Pareto goals), not
hand-written -- see the "representative search" investigation for DGDR v2
tracking issue #13545, phase-1 items 2 and 3. In particular
REAL_CANDIDATE_TEP_TRTLLM is the top-ranked scalar candidate from that run
exactly as produced, including its strategy="tep" value, which is what
surfaces the real materialization gap tested below.
"""

from __future__ import annotations

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.planner,
    pytest.mark.parallel,
]

try:
    from dynamo.profiler.v2.materializer import (
        MaterializationError,
        materialize_dgd_from_candidate,
    )
except ImportError as exc:
    pytest.skip(f"Skip (missing dependency): {exc}", allow_module_level=True)


# Verbatim from a real Sweeper.run() scalar search (see phase1-item2 fixture).
REAL_CANDIDATE_TEP_TRTLLM = {
    "deployment_mode": "agg",
    "backend": "trtllm",
    "model_name": "deepseek-ai/DeepSeek-V3",
    "hardware_sku": "gb200",
    "gpu_budget": 32,
    "min_gpu_budget": None,
    "context_length": None,
    "startup_time": None,
    "aic_nextn": None,
    "tp": 4,
    "pp": 1,
    "attention_dp": 1,
    "moe_tp": 1,
    "moe_ep": 4,
    "strategy": "tep",
    "replicas": 2,
    "used_gpus": 8,
    "agg_max_num_batched_tokens": 8192,
    "agg_max_num_seqs": 1024,
    "agg_block_size": 64,
    "agg_gpu_memory_utilization": 0.9,
    "agg_enable_prefix_caching": True,
    "backend_version": "1.3.0rc10",
    "concurrency": 64,
}

_IMAGE = "my-registry/tensorrtllm-runtime:1.3.0rc10"


def test_real_candidate_strategy_tep_on_trtllm_raises_materialization_error() -> None:
    """Confirms a real gap, not a hypothetical one.

    TrtllmConfigModifier.set_config_tep_size raises NotImplementedError
    (TEP is genuinely unsupported for this backend today). A real
    Sweeper.run() output can legally produce exactly this combination, so
    the materializer must turn this into an explicit MaterializationError
    -- not crash uninformatively and not silently fall back to a different
    strategy.
    """
    with pytest.raises(MaterializationError, match="not supported"):
        materialize_dgd_from_candidate(REAL_CANDIDATE_TEP_TRTLLM, image=_IMAGE)


def test_tp_strategy_materializes_successfully_on_all_three_backends() -> None:
    for backend in ("vllm", "sglang", "trtllm"):
        candidate = dict(REAL_CANDIDATE_TEP_TRTLLM, backend=backend, strategy="tp")
        result = materialize_dgd_from_candidate(candidate, image=_IMAGE)

        worker_components = [
            c for c in result.dgd["spec"]["components"] if c.get("type") == "worker"
        ]
        assert worker_components, f"no worker component materialized for {backend}"
        args = worker_components[0]["podTemplate"]["spec"]["containers"][0]["args"]
        assert args, f"no args materialized for {backend}"


def test_evaluation_context_fields_never_appear_in_the_dgd_spec() -> None:
    """The identity-collision-relevant fields (concurrency, kv_load_ratio,
    ...) must land in `.experimental`, never as a CLI flag in the DGD spec
    itself -- a DGD spec has no field for "what traffic this was evaluated
    against". Checks for the actual flag names a leak would produce, not a
    bare value, since a numeric value (e.g. concurrency=64) can coincide
    with an unrelated, legitimately-included field (e.g. block_size=64)."""
    candidate = dict(REAL_CANDIDATE_TEP_TRTLLM, strategy="tp")
    result = materialize_dgd_from_candidate(candidate, image=_IMAGE)

    all_args: list[str] = []
    for component in result.dgd["spec"]["components"]:
        if component.get("type") == "worker":
            all_args.extend(component["podTemplate"]["spec"]["containers"][0]["args"])

    leak_indicating_substrings = ("concurrency", "kv-load-ratio", "kv_load_ratio")
    for arg in all_args:
        for leak_marker in leak_indicating_substrings:
            assert leak_marker not in arg, f"evaluation-context flag leaked: {arg!r}"

    assert result.experimental["concurrency"] == candidate["concurrency"]
    assert result.experimental["hardware_sku"] == candidate["hardware_sku"]


def test_kv_load_ratio_pareto_candidates_carry_distinct_experimental_context() -> None:
    """Two candidates with identical deployment shape but different
    kv_load_ratio (the confirmed real Pareto-search identity-collision case)
    must still be distinguishable via `.experimental`, even though their DGD
    specs are legitimately identical."""
    base = dict(REAL_CANDIDATE_TEP_TRTLLM, strategy="tp")
    candidate_a = dict(base, kv_load_ratio=0.25, concurrency=4)
    candidate_b = dict(base, kv_load_ratio=1.0, concurrency=16)

    result_a = materialize_dgd_from_candidate(candidate_a, image=_IMAGE)
    result_b = materialize_dgd_from_candidate(candidate_b, image=_IMAGE)

    assert result_a.dgd == result_b.dgd, (
        "expected identical DGD specs for this test (that's the point -- "
        "identity must not rely on the DGD spec alone)"
    )
    assert (
        result_a.experimental["kv_load_ratio"] != result_b.experimental["kv_load_ratio"]
    )


def test_unknown_backend_raises_materialization_error() -> None:
    candidate = dict(
        REAL_CANDIDATE_TEP_TRTLLM, backend="not-a-real-backend", strategy="tp"
    )
    with pytest.raises(MaterializationError, match="no CONFIG_MODIFIERS entry"):
        materialize_dgd_from_candidate(candidate, image=_IMAGE)


def test_missing_required_field_raises_materialization_error() -> None:
    candidate = dict(REAL_CANDIDATE_TEP_TRTLLM, strategy="tp")
    del candidate["agg_block_size"]
    with pytest.raises(MaterializationError, match="missing required field"):
        materialize_dgd_from_candidate(candidate, image=_IMAGE)
