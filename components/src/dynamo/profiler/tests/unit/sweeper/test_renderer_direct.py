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
    from dynamo.profiler.sweeper.renderers.direct.materializer import (
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

# Built directly from the confirmed field-naming rules in
# aisimulate.sweeper.sample.py's unroll_sample/_unroll_parallel: disagg mode
# has NO bare tp/strategy/replicas/agg_* keys at all -- every shape/kv-cache
# /scheduler/replica/attention-dp field is uniformly "{role}_"-prefixed
# ("prefill_"/"decode_"), unlike agg's bare-vs-"agg_"-prefixed mix.
# model_name/backend/deployment_mode/hardware_sku/gpu_budget are shared,
# not role-prefixed, matching _DEPLOYMENT_PINNED.
REAL_SHAPED_DISAGG_CANDIDATE = {
    "deployment_mode": "disagg",
    "backend": "vllm",
    "model_name": "deepseek-ai/DeepSeek-V3",
    "hardware_sku": "gb200",
    "gpu_budget": 32,
    "min_gpu_budget": None,
    "context_length": None,
    "startup_time": None,
    "aic_nextn": None,
    "prefill_tp": 2,
    "prefill_pp": 1,
    "prefill_attention_dp": 1,
    "prefill_moe_tp": 1,
    "prefill_moe_ep": 1,
    "prefill_strategy": "tp",
    "prefill_replicas": 2,
    "prefill_max_num_batched_tokens": 16384,
    "prefill_max_num_seqs": 4,
    "prefill_block_size": 64,
    "prefill_gpu_memory_utilization": 0.9,
    "prefill_enable_prefix_caching": True,
    "decode_tp": 4,
    "decode_pp": 1,
    "decode_attention_dp": 1,
    "decode_moe_tp": 1,
    "decode_moe_ep": 4,
    "decode_strategy": "tep",
    "decode_replicas": 1,
    "decode_max_num_batched_tokens": 8192,
    "decode_max_num_seqs": 512,
    "decode_block_size": 64,
    "decode_gpu_memory_utilization": 0.85,
    "decode_enable_prefix_caching": False,
    "used_gpus": 8,
    "backend_version": "0.24.0",
    "concurrency": 128,
}

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


def test_evaluated_model_is_written_into_the_dgd_not_the_template_placeholder() -> None:
    """Regression test for a real bug found running a live end-to-end sweep:
    the sweep evaluated and scored Qwen/Qwen3-8B, but the materialized DGD
    contained Qwen/Qwen3-0.6B -- the base template's example placeholder --
    because materialize_dgd_from_candidate never read candidate_config
    ["model_name"] at all. Checks all three backends, since each uses a
    different flag name (vLLM: --model; SGLang/TRT-LLM: --model-path) via
    cls.WORKER_MODEL_PATH_ARG, and the bug could plausibly be fixed for one
    backend while remaining broken for the others.
    """
    for backend in ("vllm", "sglang", "trtllm"):
        candidate = dict(
            REAL_CANDIDATE_TEP_TRTLLM,
            backend=backend,
            strategy="tp",
            model_name="Qwen/Qwen3-8B",
        )
        result = materialize_dgd_from_candidate(candidate, image=_IMAGE)

        worker = next(
            c for c in result.dgd["spec"]["components"] if c.get("type") == "worker"
        )
        args = worker["podTemplate"]["spec"]["containers"][0]["args"]
        args_text = " ".join(args)

        assert "Qwen/Qwen3-8B" in args_text, (
            f"{backend}: evaluated model_name missing from materialized args: {args}"
        )
        assert "Qwen3-0.6B" not in args_text, (
            f"{backend}: template placeholder model leaked into materialized "
            f"args (the real bug this test guards against): {args}"
        )

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

def test_total_gpu_footprint_matches_the_evaluated_candidate_not_just_tp() -> None:
    """Regression test for a review-reported gap: a real 4-GPU candidate
    (Llama-3.3-70B FP8 / vLLM / 4xH200, tp=2 x replicas=2) materialized as a
    2-GPU DGD, because replicas was never propagated from candidate_config
    into component.replicas -- it silently stayed at the base template's
    hardcoded `replicas: 1` regardless of the candidate's real replica
    count. gpus-per-worker (via tp/tep/dep) was already correct; only the
    replica multiplier was missing, so total footprint (replicas x
    gpus-per-worker) could be silently wrong even when the per-worker
    GPU count looked right on its own.
    """
    candidate = dict(
        REAL_CANDIDATE_TEP_TRTLLM,
        backend="vllm",
        strategy="tp",
        tp=2,
        replicas=2,
        used_gpus=4,
        model_name="meta-llama/Llama-3.3-70B-Instruct-FP8",
    )
    result = materialize_dgd_from_candidate(candidate, image=_IMAGE)

    worker = next(
        c for c in result.dgd["spec"]["components"] if c.get("type") == "worker"
    )
    replicas = worker["replicas"]
    gpus_per_worker = int(
        worker["podTemplate"]["spec"]["containers"][0]["resources"]["limits"][
            "nvidia.com/gpu"
        ]
    )

    assert replicas == 2, f"replicas not propagated from candidate: got {replicas}"
    assert gpus_per_worker == 2
    assert replicas * gpus_per_worker == candidate["used_gpus"], (
        f"materialized total GPU footprint ({replicas} x {gpus_per_worker}) "
        f"does not match the evaluated candidate's used_gpus "
        f"({candidate['used_gpus']})"
    )

def test_scheduler_limits_materialize_the_evaluated_candidates_values() -> None:
    """Regression test for the most significant gap found in this pass:
    materialize_dgd_from_candidate never called set_prefill_config on any
    backend, so agg_max_num_seqs/agg_max_num_batched_tokens -- real search
    dimensions Sweeper evaluates and scores candidates by -- were never
    materialized at all. Every DGD silently kept the base template's fixed
    scheduler defaults regardless of what was actually searched, on every
    backend, since this materializer's very first version.
    """
    for backend in ("vllm", "sglang", "trtllm"):
        candidate = dict(
            REAL_CANDIDATE_TEP_TRTLLM,
            backend=backend,
            strategy="tp",
            agg_max_num_seqs=999,
            agg_max_num_batched_tokens=12345,
        )
        result = materialize_dgd_from_candidate(candidate, image=_IMAGE)
        worker = next(
            c for c in result.dgd["spec"]["components"] if c.get("type") == "worker"
        )
        args_text = " ".join(worker["podTemplate"]["spec"]["containers"][0]["args"])
        assert "999" in args_text, f"{backend}: agg_max_num_seqs missing: {args_text}"
        assert "12345" in args_text, (
            f"{backend}: agg_max_num_batched_tokens missing: {args_text}"
        )


def test_attention_dp_enabled_when_candidate_selected_it_trtllm_only() -> None:
    """Regression test: examples/backends/trtllm/engine_configs/qwen3/agg.yaml
    (the extra-engine-args file every TRT-LLM agg deployment references)
    hardcodes enable_attention_dp: false. Nothing overrode it before this
    fix -- a candidate genuinely evaluated with attention_dp > 1 would
    silently materialize with attention-DP off. TRT-LLM-only: vLLM/SGLang
    don't reference this file, so their modifiers correctly have no
    set_config_attention_dp at all (checked via hasattr, matching how
    materializer.py gates the call).
    """
    candidate = dict(
        REAL_CANDIDATE_TEP_TRTLLM, backend="trtllm", strategy="tp", attention_dp=4
    )
    result = materialize_dgd_from_candidate(candidate, image=_IMAGE)
    worker = next(
        c for c in result.dgd["spec"]["components"] if c.get("type") == "worker"
    )
    args = " ".join(worker["podTemplate"]["spec"]["containers"][0]["args"])
    assert "enable_attention_dp" in args and "true" in args.lower(), args

    from dynamo.profiler.utils.config_modifiers import CONFIG_MODIFIERS

    assert not hasattr(CONFIG_MODIFIERS["vllm"], "set_config_attention_dp")
    assert not hasattr(CONFIG_MODIFIERS["sglang"], "set_config_attention_dp")


def test_attention_dp_left_disabled_when_candidate_did_not_select_it() -> None:
    candidate = dict(
        REAL_CANDIDATE_TEP_TRTLLM, backend="trtllm", strategy="tp", attention_dp=1
    )
    result = materialize_dgd_from_candidate(candidate, image=_IMAGE)
    worker = next(
        c for c in result.dgd["spec"]["components"] if c.get("type") == "worker"
    )
    args = " ".join(worker["podTemplate"]["spec"]["containers"][0]["args"])
    assert "enable_attention_dp" in args and "false" in args.lower(), args


def test_cuda_graph_batch_size_mirrors_engine_max_batch_size_trtllm() -> None:
    """cuda_graph_config.max_batch_size is a separate key from top-level
    max_batch_size in the base template; confirmed it was never otherwise
    overridden and stayed fixed at the template's default (16) regardless
    of the real candidate."""
    candidate = dict(
        REAL_CANDIDATE_TEP_TRTLLM,
        backend="trtllm",
        strategy="tp",
        agg_max_num_seqs=777,
    )
    result = materialize_dgd_from_candidate(candidate, image=_IMAGE)
    worker = next(
        c for c in result.dgd["spec"]["components"] if c.get("type") == "worker"
    )
    args = worker["podTemplate"]["spec"]["containers"][0]["args"]
    args_text = " ".join(args)
    assert "cuda_graph_config.max_batch_size" in args_text
    assert "777" in args_text

def test_disagg_materializes_both_prefill_and_decode_with_their_own_shapes() -> None:
    """The core disagg test: prefill and decode must each get their own
    tp/kv-cache/scheduler/replicas from their own {role}_-prefixed fields --
    not share one shape, and not silently fall back to agg's bare/agg_*
    keys (which don't exist on a real disagg Candidate.config at all)."""
    result = materialize_dgd_from_candidate(
        REAL_SHAPED_DISAGG_CANDIDATE, image=_IMAGE
    )
    components = {
        c["type"]: c
        for c in result.dgd["spec"]["components"]
        if c.get("type") in ("prefill", "decode")
    }
    assert set(components) == {"prefill", "decode"}

    prefill_args = " ".join(
        components["prefill"]["podTemplate"]["spec"]["containers"][0]["args"]
    )
    decode_args = " ".join(
        components["decode"]["podTemplate"]["spec"]["containers"][0]["args"]
    )

    # Prefill: tp=2, replicas=2, max_num_seqs=4, deepseek-ai/DeepSeek-V3
    assert "--tensor-parallel-size 2" in prefill_args
    assert components["prefill"]["replicas"] == 2
    assert "deepseek-ai/DeepSeek-V3" in prefill_args

    # Decode: tep strategy (moe_tp=1), replicas=1, distinct from prefill
    assert components["decode"]["replicas"] == 1
    assert "deepseek-ai/DeepSeek-V3" in decode_args

    # The two roles must not have collapsed onto identical materialized args
    # -- this is the actual failure mode a broken/incomplete disagg
    # implementation could produce (e.g. accidentally reusing agg's single-
    # pass logic for both roles).
    assert prefill_args != decode_args


def test_disagg_prefix_caching_differs_correctly_by_role() -> None:
    """prefill_enable_prefix_caching=True, decode_enable_prefix_caching=False
    on the fixture -- must materialize differently per role, not share one
    value (a plausible copy-paste bug: reading only one role's field for
    both)."""
    result = materialize_dgd_from_candidate(
        REAL_SHAPED_DISAGG_CANDIDATE, image=_IMAGE
    )
    components = {
        c["type"]: c
        for c in result.dgd["spec"]["components"]
        if c.get("type") in ("prefill", "decode")
    }
    prefill_args = components["prefill"]["podTemplate"]["spec"]["containers"][0]["args"]
    decode_args = components["decode"]["podTemplate"]["spec"]["containers"][0]["args"]

    assert "--enable-prefix-caching" in prefill_args
    assert "--no-enable-prefix-caching" in decode_args


def test_disagg_missing_role_field_raises_materialization_error() -> None:
    """Confirms disagg reads {role}_-prefixed keys for real -- deleting one
    must fail, not silently fall back to agg's differently-named field."""
    candidate = dict(REAL_SHAPED_DISAGG_CANDIDATE)
    del candidate["decode_max_num_seqs"]
    with pytest.raises(MaterializationError, match="decode_max_num_seqs"):
        materialize_dgd_from_candidate(candidate, image=_IMAGE)