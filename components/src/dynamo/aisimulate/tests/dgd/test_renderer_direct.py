# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for the direct Sweeper-to-DGD materializer."""

from __future__ import annotations

import sys

import pytest

if not (3, 11) <= sys.version_info < (3, 14):
    pytest.skip("AISimulate requires Python 3.11–3.13", allow_module_level=True)

from dynamo.aisimulate.output.dgd.renderers.direct import (
    materializer as materializer_module,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.planner,
    pytest.mark.parallel,
]

MaterializationError = materializer_module.MaterializationError
materialize_dgd_from_candidate = materializer_module.materialize_dgd_from_candidate


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


@pytest.mark.parametrize(
    ("backend", "tp_flag"),
    [("vllm", "--tensor-parallel-size"), ("sglang", "--tp")],
)
def test_tep_uses_candidate_parallel_width(backend, tp_flag) -> None:
    candidate = dict(
        REAL_CANDIDATE_TEP_TRTLLM,
        backend=backend,
        model_name="Qwen/Qwen3-30B-A3B",
    )

    result = materialize_dgd_from_candidate(candidate, image=_IMAGE)

    worker = next(
        component
        for component in result.dgd["spec"]["components"]
        if component.get("type") == "worker"
    )
    container = worker["podTemplate"]["spec"]["containers"][0]
    args = container["args"]
    assert args[args.index(tp_flag) + 1] == "4"
    assert container["resources"]["limits"]["nvidia.com/gpu"] == "4"


def test_tp_larger_than_one_node_materializes_multinode_worker() -> None:
    candidate = dict(
        REAL_CANDIDATE_TEP_TRTLLM,
        backend="vllm",
        strategy="tp",
        tp=16,
        replicas=1,
        used_gpus=16,
    )

    result = materialize_dgd_from_candidate(
        candidate, image=_IMAGE, num_gpus_per_node=8
    )

    worker = next(
        component
        for component in result.dgd["spec"]["components"]
        if component.get("type") == "worker"
    )
    container = worker["podTemplate"]["spec"]["containers"][0]
    assert container["resources"]["limits"]["nvidia.com/gpu"] == "8"
    assert worker["multinode"] == {"nodeCount": 2}


def test_vllm_fixed_kv_capacity_uses_num_gpu_blocks_override() -> None:
    candidate = dict(
        REAL_CANDIDATE_TEP_TRTLLM,
        backend="vllm",
        strategy="tp",
        agg_gpu_memory_utilization=None,
        agg_num_gpu_blocks=512,
    )

    result = materialize_dgd_from_candidate(candidate, image=_IMAGE)

    worker = next(
        component
        for component in result.dgd["spec"]["components"]
        if component.get("type") == "worker"
    )
    args = worker["podTemplate"]["spec"]["containers"][0]["args"]
    assert "--gpu-memory-utilization" not in args
    index = args.index("--num-gpu-blocks-override")
    assert args[index + 1] == "512"


@pytest.mark.parametrize(
    (
        "backend",
        "model_flag",
        "sequence_flag",
        "tokens_flag",
        "memory_flag",
        "expected_fraction",
    ),
    [
        (
            "vllm",
            "--model",
            "--max-num-seqs",
            "--max-num-batched-tokens",
            "--gpu-memory-utilization",
            "0.9",
        ),
        (
            "sglang",
            "--model-path",
            "--max-running-requests",
            "--chunked-prefill-size",
            "--mem-fraction-static",
            "0.88",
        ),
        (
            "trtllm",
            "--model-path",
            "--trtllm.max_batch_size",
            "--trtllm.max_num_tokens",
            "--trtllm.kv_cache_config.free_gpu_memory_fraction",
            "0.9",
        ),
    ],
)
def test_aggregated_candidate_preserves_engine_configuration(
    backend, model_flag, sequence_flag, tokens_flag, memory_flag, expected_fraction
) -> None:
    candidate = dict(
        REAL_CANDIDATE_TEP_TRTLLM,
        backend=backend,
        strategy="tp",
        model_name="Qwen/Qwen3-8B",
        tp=2,
        replicas=2,
        used_gpus=4,
        agg_max_num_seqs=999,
        agg_max_num_batched_tokens=12345,
        agg_gpu_memory_utilization=None,
    )

    result = materialize_dgd_from_candidate(candidate, image=_IMAGE)

    worker = next(
        component
        for component in result.dgd["spec"]["components"]
        if component.get("type") == "worker"
    )
    container = worker["podTemplate"]["spec"]["containers"][0]
    args = container["args"]
    for flag, value in (
        (model_flag, candidate["model_name"]),
        ("--served-model-name", candidate["model_name"]),
        (sequence_flag, "999"),
        (tokens_flag, "12345"),
        (memory_flag, expected_fraction),
    ):
        assert args.count(flag) == 1
        assert args[args.index(flag) + 1] == value
    if backend == "trtllm":
        assert (
            args[args.index("--trtllm.cuda_graph_config.max_batch_size") + 1] == "999"
        )
    assert worker["replicas"] == 2
    assert int(container["resources"]["limits"]["nvidia.com/gpu"]) == 2
    assert "Qwen3-0.6B" not in " ".join(args)


def test_kv_capacity_rejects_memory_fraction_and_fixed_blocks_together() -> None:
    candidate = dict(
        REAL_CANDIDATE_TEP_TRTLLM,
        backend="vllm",
        strategy="tp",
        agg_num_gpu_blocks=512,
    )

    with pytest.raises(MaterializationError, match="cannot set both"):
        materialize_dgd_from_candidate(candidate, image=_IMAGE)


@pytest.mark.parametrize("backend", ["sglang", "trtllm"])
def test_fixed_kv_capacity_fails_when_backend_cannot_materialize_it(backend) -> None:
    candidate = dict(
        REAL_CANDIDATE_TEP_TRTLLM,
        backend=backend,
        strategy="tp",
        agg_gpu_memory_utilization=None,
        agg_num_gpu_blocks=512,
    )

    with pytest.raises(MaterializationError, match="fixed KV-cache blocks"):
        materialize_dgd_from_candidate(candidate, image=_IMAGE)


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


def test_modifier_value_error_is_reported_as_materialization_error(monkeypatch) -> None:
    def fail_materialization(*_args, **_kwargs):
        raise ValueError("invalid modifier input")

    monkeypatch.setattr(
        materializer_module,
        "_materialize_worker",
        fail_materialization,
    )
    candidate = dict(REAL_CANDIDATE_TEP_TRTLLM, strategy="tp")

    with pytest.raises(MaterializationError, match="materialization failed"):
        materialize_dgd_from_candidate(candidate, image=_IMAGE)


@pytest.mark.parametrize("attention_dp,expected", [(1, "false"), (4, "true")])
def test_trtllm_attention_dp_matches_evaluated_candidate(
    attention_dp, expected
) -> None:
    candidate = dict(
        REAL_CANDIDATE_TEP_TRTLLM,
        backend="trtllm",
        strategy="tp",
        attention_dp=attention_dp,
    )
    result = materialize_dgd_from_candidate(candidate, image=_IMAGE)
    worker = next(
        c for c in result.dgd["spec"]["components"] if c.get("type") == "worker"
    )
    args = worker["podTemplate"]["spec"]["containers"][0]["args"]
    assert args[args.index("--trtllm.enable_attention_dp") + 1] == expected


def test_disagg_preserves_role_specific_configuration() -> None:
    result = materialize_dgd_from_candidate(REAL_SHAPED_DISAGG_CANDIDATE, image=_IMAGE)
    components = {
        c["type"]: c
        for c in result.dgd["spec"]["components"]
        if c.get("type") in ("prefill", "decode")
    }
    assert set(components) == {"prefill", "decode"}

    prefill_args = components["prefill"]["podTemplate"]["spec"]["containers"][0]["args"]
    decode_args = components["decode"]["podTemplate"]["spec"]["containers"][0]["args"]

    for role, args in (("prefill", prefill_args), ("decode", decode_args)):
        for flag, field in (
            ("--tensor-parallel-size", "tp"),
            ("--max-num-seqs", "max_num_seqs"),
            ("--max-num-batched-tokens", "max_num_batched_tokens"),
            ("--block-size", "block_size"),
            ("--gpu-memory-utilization", "gpu_memory_utilization"),
        ):
            assert args[args.index(flag) + 1] == str(
                REAL_SHAPED_DISAGG_CANDIDATE[f"{role}_{field}"]
            )
        assert (
            args[args.index("--model") + 1]
            == REAL_SHAPED_DISAGG_CANDIDATE["model_name"]
        )
        assert (
            components[role]["replicas"]
            == REAL_SHAPED_DISAGG_CANDIDATE[f"{role}_replicas"]
        )
    assert (
        components["decode"]["podTemplate"]["spec"]["containers"][0]["resources"][
            "limits"
        ]["nvidia.com/gpu"]
        == "4"
    )
    assert "--enable-prefix-caching" in prefill_args
    assert "--no-enable-prefix-caching" in decode_args


def test_disagg_missing_role_field_raises_materialization_error() -> None:
    """Confirms disagg reads {role}_-prefixed keys for real -- deleting one
    must fail, not silently fall back to agg's differently-named field."""
    candidate = dict(REAL_SHAPED_DISAGG_CANDIDATE)
    del candidate["decode_max_num_seqs"]
    with pytest.raises(MaterializationError, match="decode_max_num_seqs"):
        materialize_dgd_from_candidate(candidate, image=_IMAGE)
