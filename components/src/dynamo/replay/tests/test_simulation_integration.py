# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Optional-dependency preflight must run before the simulation imports.
# ruff: noqa: E402

"""Real Replay integration coverage for the Sweeper adapter/runner boundary."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip(
    "aisimulate.sweeper",
    reason="AI Simulate is an optional Dynamo simulation dependency",
)

from aisimulate.sweeper.config import OptimizationTarget, SmartSearchConfig
from aisimulate.sweeper.deploy import build_backend_deployment
from aisimulate.sweeper.kv_estimate import resolve_backend_version
from aisimulate.sweeper.provider import (
    AdapterReplaySpec,
    CandidateContext,
    RuntimeHookSpec,
    SweepContext,
)
from aisimulate.sweeper.replay import (
    BackendDeploymentSpec,
    ReplayOutputRequirements,
    ReplaySpec,
)
from aisimulate.sweeper.sample import unroll_sample
from aisimulate.sweeper.sampler import Suggestion
from aisimulate.sweeper.score import objective_value
from aisimulate.sweeper.search import Sweeper
from aisimulate.sweeper.search_space import enumerate_branches

from dynamo.mocker import MockEngineArgs
from dynamo.planner.simulation import create_provider as create_planner_provider
from dynamo.replay import run_trace_replay
from dynamo.replay.simulation import DynamoReplayRunnerFactory
from dynamo.router.simulation import create_provider as create_router_provider

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.integration,
    pytest.mark.planner,
    pytest.mark.filterwarnings("ignore:invalid escape sequence.*:SyntaxWarning"),
    pytest.mark.filterwarnings("ignore:invalid escape sequence.*:DeprecationWarning"),
]

_TRACE = str(Path(__file__).resolve().parent / "data/mooncake_tiny.jsonl")


def _config(
    *,
    scaling_policy: str,
    candidates_per_round: int = 1,
    parallel_evals: int = 1,
    include_router: bool = False,
) -> SmartSearchConfig:
    # Replay consumes model metadata and the AIC performance database; it does
    # not load model weights or require a GPU. This model/backend pair is kept
    # because it has complete coverage in the GB200 performance database.
    adapters = {
        "dynamo.planner": {
            "search_space": {
                "scaling_policy": {"preset": [scaling_policy]},
                "fpm_sampling": {"preset": ["default"]},
                "load_sensitivity": {"preset": ["default"]},
                "load_predictor": {"preset": ["constant_last"]},
            }
        }
    }
    if include_router:
        adapters["dynamo.router"] = {
            "search_space": {
                "mode": ["kv_router"],
                "overlap_score_credit": [0.5],
                "prefill_load_scale": [1.0],
                "temperature": [0.2],
            }
        }

    return SmartSearchConfig(
        search_space={
            "model_name": "meta-llama/Meta-Llama-3.1-8B",
            "hardware_sku": "gb200",
            "backend": ["trtllm"],
            "deployment_mode": ["agg"],
            "gpu_budget": 256,
        },
        adapters=adapters,
        # Slow the trace enough for load_180_5 to execute a Planner tick.
        workload={"trace_path": _TRACE, "arrival_speedup_ratio": 0.5},
        sweep={
            "max_rounds": 1,
            "candidates_per_round": candidates_per_round,
            "parallel_evals": parallel_evals,
            "max_eval_seconds": 240,
        },
        goal={
            "target": "goodput_per_gpu",
            "sla": {"ttft_ms": 8000.0, "itl_ms": 200.0},
        },
    )


class _TwoCandidateSampler:
    """Choose two deterministic valid candidates without depending on Vizier."""

    def __init__(self, branch, study_id, objectives=None):
        del study_id, objectives
        self.branch = branch

    def suggest(self, count):
        assert count == 2
        suggestions = []
        for index in range(count):
            selection = {
                name: values[min(index, len(values) - 1)]
                for name, values in self.branch.knob_choices.items()
            }
            selection["deployment_mode"] = self.branch.deployment_mode
            suggestions.append(
                Suggestion(
                    selection=selection,
                    parallel_config=self.branch.parallel_configs[
                        min(index, len(self.branch.parallel_configs) - 1)
                    ],
                    handle=index,
                )
            )
        return suggestions

    def observe(self, suggestion, metrics):
        del suggestion, metrics

    def observe_infeasible(self, suggestion, reason):
        pytest.fail(f"unexpected infeasible suggestion {suggestion}: {reason}")


def _run_one(policy: str):
    config = _config(scaling_policy=policy)
    runner_factory = DynamoReplayRunnerFactory()
    branch = enumerate_branches(
        config,
        runner_capabilities=runner_factory.capabilities(),
    )[0]
    parallel_config = branch.parallel_configs[0]
    selection = {
        "deployment_mode": "agg",
        "backend": "trtllm",
        "agg_max_num_batched_tokens": 16384,
        "agg_max_num_seqs": 512,
    }
    sample = unroll_sample(
        search_space=config.search_space,
        selection=selection,
        parallel_config=parallel_config,
    )
    backend_version = resolve_backend_version("gb200", "trtllm")
    sample["backend_version"] = backend_version
    backend_deployment = build_backend_deployment(
        sample,
        backend_version=backend_version,
    )

    adapter = create_planner_provider()
    adapter_plan = adapter.generate_search_space(
        config.adapters["dynamo.planner"].search_space,
        SweepContext(
            core_search_space=config.search_space.model_dump(mode="json"),
            workload=config.workload.model_dump(mode="json"),
            goal=config.goal.model_dump(mode="json"),
            show_progress=False,
        ),
    )
    adapter_selection = {"scaling_policy": policy}
    if policy != "disabled":
        adapter_selection.update(
            fpm_sampling="default",
            load_sensitivity="default",
        )
    adapter_spec = adapter.materialize_replay(
        adapter_plan,
        adapter_selection,
        CandidateContext(
            sample=sample,
            backend_deployment=backend_deployment,
        ),
    )
    replay_spec = ReplaySpec(
        backend_deployment=backend_deployment,
        workload=config.workload.model_dump(mode="json"),
        goal=config.goal.model_dump(mode="json"),
        adapters={"dynamo.planner": adapter_spec},
    )

    runner = runner_factory.create(0)
    try:
        return runner.run(replay_spec), adapter_spec
    finally:
        runner.close()


@pytest.mark.pre_merge
@pytest.mark.timeout(300)
def test_real_planner_bridge_preserves_goodput_and_gpu_hours() -> None:
    report, adapter_spec = _run_one("load_180_5")

    assert adapter_spec.runtime_hooks
    assert report.metrics["goodput_output_throughput_tok_s"] > 0.0
    assert report.metrics["gpu_hours"] > 0.0
    assert report.metrics["planner_total_ticks"] >= 1
    avg_gpu = report.metrics["gpu_hours"] / (
        report.metrics["duration_ms"] / 3_600_000.0
    )
    expected = report.metrics["goodput_output_throughput_tok_s"] / avg_gpu
    assert objective_value(
        report.metrics,
        OptimizationTarget.GOODPUT_PER_GPU,
    ) == pytest.approx(expected)


@pytest.mark.pre_merge
@pytest.mark.timeout(300)
def test_real_static_path_preserves_goodput() -> None:
    report, adapter_spec = _run_one("disabled")

    assert adapter_spec.runtime_hooks == ()
    assert report.metrics["goodput_output_throughput_tok_s"] > 0.0
    assert report.metrics["gpu_hours"] > 0.0
    assert "planner_total_ticks" not in report.metrics


@pytest.mark.pre_merge
@pytest.mark.post_merge
@pytest.mark.timeout(300)
def test_sweeper_runs_real_dynamo_replay_in_spawned_workers() -> None:
    config = _config(
        scaling_policy="load_180_5",
        candidates_per_round=2,
        parallel_evals=2,
        include_router=True,
    )

    result = Sweeper(
        runner_factory=DynamoReplayRunnerFactory(),
        providers={
            "dynamo.planner": create_planner_provider(),
            "dynamo.router": create_router_provider(),
        },
        sampler_factory=_TwoCandidateSampler,
        show_progress=False,
    ).run(config)

    candidates = result.candidates
    assert len(candidates) == 2
    assert all(candidate.status == "feasible" for candidate in candidates)
    assert all(
        candidate.metrics["output_throughput_tok_s"] > 0.0 for candidate in candidates
    )
    assert all(candidate.metrics["gpu_hours"] > 0.0 for candidate in candidates)
    assert all(
        candidate.metrics["planner_total_ticks"] >= 1 for candidate in candidates
    )
    assert all(
        candidate.config["adapters"]["dynamo.router"]["mode"] == "kv_router"
        for candidate in candidates
    )


def _fixed_engine_args(backend: str, role: str, dp_size: int = 1) -> dict:
    return {
        "engine_type": backend,
        "worker_type": role,
        "dp_size": dp_size,
        "block_size": 4,
        "num_gpu_blocks": 64,
        "timing_model": {"type": "fixed", "prefill_ms": 1.0, "decode_ms": 1.0},
    }


def _aic_engine_args(backend: str, role: str, dp_size: int = 1) -> dict:
    args = _fixed_engine_args(backend, role, dp_size)
    model = Path(__file__).parent / "e2e/configs/unified_cli/fixtures/tiny-model"
    args["timing_model"] = {
        "type": "external",
        "provider": "aic",
        "config": {
            "model": str(model.resolve()),
            "system": "h200_sxm",
            "backend": backend,
            "backend_version": "0.24.0" if backend == "vllm" else "0.5.14",
            "worker_type": role,
            "tp": dp_size,
            "attention_dp": dp_size,
            "estimation_mode": "op_level",
        },
    }
    return args


@pytest.mark.pre_merge
@pytest.mark.timeout(30)
@pytest.mark.parametrize("backend", ["vllm", "sglang", "trtllm"])
@pytest.mark.parametrize("prefill_dp,decode_dp", [(2, 1), (1, 2), (2, 4)])
def test_real_runner_supports_disaggregated_attention_dp(
    backend: str, prefill_dp: int, decode_dp: int
) -> None:
    factory = DynamoReplayRunnerFactory()
    assert factory.capabilities().supports_attention_dp("disagg", prefill_dp, decode_dp)
    spec = ReplaySpec(
        backend_deployment=BackendDeploymentSpec(
            deployment_mode="disagg",
            backend=backend,
            backend_version="current",
            prefill_engine_args=_fixed_engine_args(backend, "prefill", prefill_dp),
            decode_engine_args=_fixed_engine_args(backend, "decode", decode_dp),
            num_prefill_workers=1,
            num_decode_workers=1,
        ),
        workload={"isl": 16, "osl": 2, "request_count": 8, "concurrency": 4},
        goal={"target": "goodput", "sla": {"ttft_ms": 1000.0, "itl_ms": 1000.0}},
    )
    runner = factory.create(0)
    try:
        report = runner.run(
            spec, output_requirements=ReplayOutputRequirements(capture_per_request=True)
        )
    finally:
        runner.close()

    assert report.metrics["completed_requests"] == 8
    assert report.metrics["goodput_output_throughput_tok_s"] > 0
    records = report.metadata["native_report"]["per_request"]
    assert len(records) == 8
    for role, dp_size in [("prefill", prefill_dp), ("decode", decode_dp)]:
        routes = [
            route
            for record in records
            for route in record["routing_history"]
            if route["pool"] == role
        ]
        assert len(routes) == 8
        assert {(route["logical_worker_id"], route["dp_rank"]) for route in routes} == {
            (0, rank) for rank in range(dp_size)
        }


@pytest.mark.pre_merge
@pytest.mark.timeout(30)
@pytest.mark.parametrize("backend", ["vllm", "sglang"])
@pytest.mark.parametrize("router_mode", ["round_robin", "kv_router"])
@pytest.mark.parametrize(
    "speculative",
    ["off", "public", "legacy", "native", "canonical", "canonical_legacy"],
)
def test_real_runner_preserves_disaggregated_agentic_dependencies(
    backend: str, router_mode: str, speculative: str
) -> None:
    trace = (
        Path(__file__).parent
        / "e2e/configs/unified_cli/fixtures/traces/agentic-mooncake.jsonl"
    )
    decode_engine_args = (
        _fixed_engine_args(backend, "decode", 4)
        if speculative == "off"
        else _aic_engine_args(backend, "decode", 4)
    )
    if speculative == "public":
        decode_engine_args["speculation"] = {
            "kind": "mtp",
            "num_speculative_tokens": 3,
            "expected_accepted_tokens": 2.4,
            "seed": 42,
        }
    elif speculative == "legacy":
        decode_engine_args.update(aic_nextn=3, aic_nextn_accepted=2.4, aic_mtp_seed=42)
    elif speculative == "native":
        decode_engine_args.update(
            ais_nextn=3, ais_nextn_accept_rates="1,1,0.4", ais_mtp_seed=42
        )
    elif speculative in {"canonical", "canonical_legacy"}:
        config = decode_engine_args.pop("timing_model")["config"]
        if speculative == "canonical":
            config["speculation"] = {
                "kind": "mtp",
                "params": {"num_speculative_tokens": 3},
            }
        else:
            config["nextn"] = 3
        decode_engine_args.update(
            ais_perf_config=config,
            ais_nextn_accept_rates="1,1,0.4",
            ais_mtp_seed=42,
        )
    spec = ReplaySpec(
        backend_deployment=BackendDeploymentSpec(
            deployment_mode="disagg",
            backend=backend,
            backend_version="0.24.0" if backend == "vllm" else "0.5.14",
            prefill_engine_args=_fixed_engine_args(backend, "prefill", 2),
            decode_engine_args=decode_engine_args,
            num_prefill_workers=1,
            num_decode_workers=1,
            performance_model_metadata={
                role: {
                    "config": {
                        "model": str(
                            (
                                Path(__file__).parent
                                / "e2e/configs/unified_cli/fixtures/tiny-model"
                            ).resolve()
                        )
                    }
                }
                for role in ("prefill", "decode")
            },
        ),
        workload={
            "trace_path": str(trace),
            "trace_format": "agentic_mooncake",
            "trace_block_size": 4,
            "agentic_lanes": 1,
        },
        goal={"target": "throughput"},
        adapters={
            "dynamo.router": AdapterReplaySpec(
                runtime_hooks=(
                    RuntimeHookSpec(
                        provider="dynamo.router",
                        kind="placement_policy",
                        api_version=1,
                        config={"router_mode": router_mode, "router_config": {}},
                    ),
                )
            )
        },
    )
    runner = DynamoReplayRunnerFactory().create(0)
    try:
        report = runner.run(
            spec, output_requirements=ReplayOutputRequirements(capture_per_request=True)
        )
    finally:
        runner.close()

    assert report.metrics["completed_requests"] == 3
    assert report.metrics["total_output_tokens"] == 6
    assert report.metrics["completed_trajectories"] == 1
    assert report.metrics["incomplete_trajectories"] == 0
    assert report.metadata["agentic_qualification"] == "functional_only"
    if speculative != "off":
        acceptance = report.metadata["speculative_acceptance"]
        assert acceptance["decode_forwards"] == 3
        # Acceptance counts the sampled drafts plus a base token before
        # request completion clips the burst to the remaining output budget.
        assert 9 <= acceptance["accepted_tokens_including_base"] <= 12
        assert 3 <= acceptance["mean_accept_length"] <= 4
        assumptions = report.metadata["speculation"]["decode"]
        assert assumptions["expected_accepted_draft_tokens"] == (
            2.4 if speculative in {"public", "legacy"} else None
        )
        assert list(
            map(float, assumptions["conditional_acceptance_rates"].split(","))
        ) == pytest.approx([1, 1, 0.4])
        assert assumptions["seed"] == 42
        assert assumptions["resolved_method"] == "mtp"
        if speculative == "canonical":
            assert assumptions["requested"] == {
                "kind": "mtp",
                "params": {"num_speculative_tokens": 3},
            }
    records = {
        record["request_id"]: record
        for record in report.metadata["native_report"]["per_request"]
    }
    root, child, join = (
        records[name] for name in ("agent-root", "agent-child", "agent-join")
    )
    assert child["dispatched_at_ms"] == pytest.approx(root["dispatched_at_ms"] + 5)
    assert join["dispatched_at_ms"] == pytest.approx(
        max(root["terminal_time_ms"], child["terminal_time_ms"]) + 1
    )


@pytest.mark.pre_merge
@pytest.mark.timeout(60)
@pytest.mark.parametrize("backend", ["vllm", "sglang"])
@pytest.mark.parametrize("router_mode", ["round_robin", "kv_router"])
@pytest.mark.parametrize("deployment_mode", ["agg", "disagg"])
def test_real_agentic_mtp_prices_full_bursts(
    tmp_path, backend, router_mode, deployment_mode
):
    source = (
        Path(__file__).parent
        / "e2e/configs/unified_cli/fixtures/traces/agentic-mooncake.jsonl"
    )
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    for row in rows[1:]:
        row["output_length"] = 32
    trace = tmp_path / "agentic-long-output.jsonl"
    trace.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    role = "aggregated" if deployment_mode == "agg" else "decode"
    args = _aic_engine_args(backend, role)
    model = args["timing_model"]["config"]["model"]
    if deployment_mode == "agg":
        engines = {"agg_engine_args": args, "num_workers": 2}
        roles = ("aggregated",)
    else:
        engines = {
            "prefill_engine_args": _aic_engine_args(backend, "prefill"),
            "decode_engine_args": args,
            "num_prefill_workers": 1,
            "num_decode_workers": 2,
        }
        roles = ("prefill", "decode")
    spec = ReplaySpec(
        backend_deployment=BackendDeploymentSpec(
            deployment_mode=deployment_mode,
            backend=backend,
            backend_version="0.24.0" if backend == "vllm" else "0.5.14",
            performance_model_metadata={
                name: {"config": {"model": model}} for name in roles
            },
            **engines,
        ),
        workload={
            "trace_path": str(trace),
            "trace_format": "agentic_mooncake",
            "trace_block_size": 4,
            "agentic_lanes": 1,
        },
        goal={"target": "throughput"},
        adapters={
            "dynamo.router": AdapterReplaySpec(
                runtime_hooks=(
                    RuntimeHookSpec(
                        provider="dynamo.router",
                        kind="placement_policy",
                        api_version=1,
                        config={"router_mode": router_mode, "router_config": {}},
                    ),
                )
            )
        },
    )
    speculative = {
        **args,
        "speculation": {
            "kind": "mtp",
            "num_speculative_tokens": 2,
            "expected_accepted_tokens": 1.5,
            "seed": 42,
        },
    }
    field = "agg_engine_args" if deployment_mode == "agg" else "decode_engine_args"
    speculative_spec = replace(
        spec,
        backend_deployment=replace(spec.backend_deployment, **{field: speculative}),
    )
    output = ReplayOutputRequirements(capture_per_request=True)
    runner = DynamoReplayRunnerFactory().create(0)
    try:
        baseline = runner.run(spec, output_requirements=output)
        report = runner.run(speculative_spec, output_requirements=output)
    finally:
        runner.close()
    for result in (baseline, report):
        assert result.metrics["completed_requests"] == 3
        assert result.metrics["total_output_tokens"] == 96
        assert result.metrics["completed_trajectories"] == 1
        assert result.metrics["incomplete_trajectories"] == 0
        assert result.metrics["duration_ms"] > 0
        assert result.metadata["agentic_qualification"] == "functional_only"
        records = result.metadata["native_report"]["per_request"]
        assert len(records) == 3
        assert all(record["output_length"] == 32 for record in records)
    assert "speculation" not in baseline.metadata
    baseline_acceptance = baseline.metadata["speculative_acceptance"]
    # Aggregated prefill emits the first token; disaggregated prefill hands
    # that work to the decode pool, which generates all 32 output tokens.
    assert baseline_acceptance["decode_forwards"] == (
        93 if deployment_mode == "agg" else 96
    )
    assert baseline_acceptance["mean_accept_length"] == 1
    acceptance = report.metadata["speculative_acceptance"]
    assert 3 < acceptance["decode_forwards"] < baseline_acceptance["decode_forwards"]
    assert 2 <= acceptance["mean_accept_length"] <= 3
    assert report.metrics["duration_ms"] < baseline.metrics["duration_ms"]
    assumptions = report.metadata["speculation"][role]
    assert assumptions["num_speculative_tokens"] == 2
    assert assumptions["expected_accepted_draft_tokens"] == 1.5
    assert assumptions["resolved_method"] == "mtp"
    assert (
        assumptions["cost_approximation"]
        == "op_level_nextn_target_layers_and_depth_plus_one_verification"
    )
    assert assumptions["target_model"] == model


@pytest.mark.pre_merge
@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "missing", ["capacity", "rates", "aic_cost", "none", "rates_overridden"]
)
@pytest.mark.parametrize("construction", ["json", "constructor"])
def test_native_auto_agentic_trace_enforces_speculative_assumptions(
    missing, construction
):
    args = _aic_engine_args("vllm", "aggregated")
    args["ais_perf_config"] = args.pop("timing_model")["config"]
    args["ais_perf_config"]["nextn"] = 2
    args["ais_nextn_accept_rates"] = "1,0.5"
    expected = {
        "capacity": "explicit.*num_gpu_blocks",
        "rates": "explicit.*ais_nextn_accept_rates",
        "aic_cost": "op_level",
    }
    if missing == "capacity":
        args.pop("num_gpu_blocks")
    elif missing in {"rates", "rates_overridden"}:
        args.pop("ais_nextn_accept_rates")
    elif missing == "aic_cost":
        args.pop("ais_perf_config")
        args["ais_nextn"] = 2
        args["timing_model"] = {"type": "fixed", "prefill_ms": 1, "decode_ms": 1}
    else:
        args = _fixed_engine_args("vllm", "aggregated")
    if construction == "json":
        native = MockEngineArgs.from_json(json.dumps(args))
    else:
        # The constructor's default non-AIC timing is also inadmissible for SD.
        args.pop("timing_model", None)
        native = MockEngineArgs(**args)
    if missing == "rates":
        native = native.with_overrides(ais_mtp_seed=43)
    elif missing == "rates_overridden":
        native = native.with_overrides(ais_nextn_accept_rates="1,0.5")
    trace = (
        Path(__file__).parent
        / "e2e/configs/unified_cli/fixtures/traces/dynamo-agentic.jsonl"
    )
    kwargs = dict(
        trace_files=str(trace),
        trace_format="dynamo",
        trace_block_size=4,
        extra_engine_args=native,
        execution_model="target-model",
    )
    if missing in {"none", "rates_overridden"}:
        report = run_trace_replay(**kwargs)
        assert report.summary["completed_requests"] == 4
    else:
        with pytest.raises(Exception, match=expected[missing]):
            run_trace_replay(**kwargs)


@pytest.mark.pre_merge
@pytest.mark.timeout(30)
@pytest.mark.parametrize("capacity", ["missing", "inferred", "explicit"])
def test_runner_auto_agentic_retains_authored_capacity_provenance(capacity):
    args = _aic_engine_args("vllm", "aggregated")
    args["speculation"] = {
        "kind": "mtp",
        "num_speculative_tokens": 2,
        "expected_accepted_tokens": 1.5,
        "seed": 42,
    }
    if capacity == "missing":
        args.pop("num_gpu_blocks")
    elif capacity == "inferred":
        args["num_gpu_blocks_is_explicit"] = False
    spec = ReplaySpec(
        backend_deployment=BackendDeploymentSpec(
            deployment_mode="agg",
            backend="vllm",
            backend_version="0.24.0",
            agg_engine_args=args,
            num_workers=1,
        ),
        workload={
            "trace_path": str(
                Path(__file__).parent
                / "e2e/configs/unified_cli/fixtures/traces/dynamo-agentic.jsonl"
            ),
            "trace_format": "dynamo",
        },
        goal={},
    )
    runner = DynamoReplayRunnerFactory().create(0)
    try:
        if capacity != "explicit":
            with pytest.raises(
                Exception, match="explicitly configured positive num_gpu_blocks"
            ):
                runner.run(spec)
        else:
            report = runner.run(spec)
            assert report.metrics["completed_requests"] == 4
            assert report.metadata["agentic_qualification"] == "functional_only"
            assert (
                report.metadata["speculation"]["aggregated"]["capacity_source"]
                == "explicit_fixed"
            )
    finally:
        runner.close()


@pytest.mark.pre_merge
@pytest.mark.timeout(30)
@pytest.mark.parametrize("capacity", ["missing", "inferred", "explicit"])
def test_direct_loader_preserves_auto_agentic_capacity_provenance(capacity):
    from dynamo.replay.config import load_engine_args

    args = _aic_engine_args("vllm", "aggregated")
    args["ais_perf_config"] = args.pop("timing_model")["config"]
    args["ais_perf_config"]["speculation"] = {
        "kind": "mtp",
        "params": {"num_speculative_tokens": 2},
    }
    args["ais_nextn_accept_rates"] = "1,0.5"
    if capacity == "missing":
        args.pop("num_gpu_blocks")
    elif capacity == "inferred":
        args["num_gpu_blocks_is_explicit"] = False
    engine = load_engine_args(args)
    kwargs = dict(
        trace_files=str(
            Path(__file__).parent
            / "e2e/configs/unified_cli/fixtures/traces/dynamo-agentic.jsonl"
        ),
        trace_format="dynamo",
        extra_engine_args=engine,
        execution_model="target-model",
    )
    if capacity != "explicit":
        with pytest.raises(
            Exception, match="explicitly configured positive num_gpu_blocks"
        ):
            run_trace_replay(**kwargs)
    else:
        assert run_trace_replay(**kwargs).summary["completed_requests"] == 4
