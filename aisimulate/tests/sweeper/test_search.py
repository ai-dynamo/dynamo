# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Core Sweeper orchestration parity through the RunnerFactory/ReplaySpec boundary."""

from pathlib import Path

import pytest

import aisimulate.sweeper.search as search_mod
from aisimulate.sweeper.config import OptimizationGoal, SmartSearchConfig
from aisimulate.sweeper.kv_load import KVLoadResolution
from aisimulate.sweeper.parallel_enum import ParallelShape, ReplicaParallelConfig
from aisimulate.sweeper.replay import (
    BackendDeploymentSpec,
    ReplayReport,
    ReplaySpec,
    RunnerCapabilities,
)
from aisimulate.sweeper.sampler import Suggestion
from aisimulate.sweeper.search import Sweeper
from aisimulate.sweeper.search_space import BranchSpace

TRACE = str(Path(__file__).parent / "data" / "mooncake_tiny.jsonl")


def _config(gpu_budget=32, **sweep_overrides):
    sweep = {
        "max_rounds": 1,
        "candidates_per_round": 3,
        "parallel_evals": 1,
    }
    sweep.update(sweep_overrides)
    return SmartSearchConfig(
        search_space={
            "model_name": "deepseek-ai/DeepSeek-V3",
            "hardware_sku": "gb200",
            "backend": ["trtllm"],
            "deployment_mode": ["agg"],
            "gpu_budget": gpu_budget,
        },
        workload={"trace_path": TRACE},
        sweep=sweep,
        goal={"target": "throughput"},
    )


def _run_sweep(
    config: SmartSearchConfig,
    *,
    runner_factory,
    sampler_factory,
    show_progress: bool,
    on_round=None,
):
    return Sweeper(
        runner_factory=runner_factory,
        sampler_factory=sampler_factory,
        show_progress=show_progress,
    ).run(config, on_round=on_round)


def _selection(seqs: int) -> dict:
    return {
        "deployment_mode": "agg",
        "backend": "trtllm",
        "agg_max_num_batched_tokens": 8192,
        "agg_max_num_seqs": seqs,
    }


class _FakeSampler:
    """Suggest ``count`` backend candidates with increasing max_num_seqs."""

    def __init__(self, branch, study_id, objectives=None):
        self.branch = branch
        self.objectives = objectives
        self.scored: list = []

    def suggest(self, count):
        return [
            Suggestion(
                selection=(selection := _selection(256 * (i + 1))),
                parallel_config=self.branch.parallel_configs[0],
                handle=selection,
            )
            for i in range(count)
        ]

    def observe(self, suggestion, metrics):
        self.scored.append(metrics)

    def observe_infeasible(self, suggestion, reason):
        self.scored.append(("infeasible", reason))


class _FakeRunner:
    """Throughput equals max_num_seqs, making ranking deterministic."""

    def __init__(self):
        self.calls = 0
        self.specs: list[ReplaySpec] = []
        self.closed = False

    def run(self, spec: ReplaySpec) -> ReplayReport:
        self.calls += 1
        self.specs.append(spec)
        args = spec.backend_deployment.agg_engine_args
        assert args is not None
        return ReplayReport(
            metrics={
                "output_throughput_tok_s": float(args["max_num_seqs"]),
                "gpu_hours": 1.0,
            }
        )

    def close(self):
        self.closed = True


class _FakeRunnerFactory:
    def __init__(self, runner=None, *, topologies=(("*", "*"),)):
        self.runner = runner or _FakeRunner()
        self.topologies = topologies
        self.worker_ids: list[int] = []

    def capabilities(self):
        return RunnerCapabilities(supported_backend_topologies=self.topologies)

    def create(self, worker_id):
        self.worker_ids.append(worker_id)
        return self.runner


def _branch(parallel_config):
    return BranchSpace(
        deployment_mode="agg",
        parallel_configs=(parallel_config,),
        supported_backends={parallel_config: frozenset({"trtllm"})},
        knob_choices={"backend": ["trtllm"]},
    )


def _stub(monkeypatch, branch):
    monkeypatch.setattr(
        search_mod,
        "enumerate_branches",
        lambda config, *, max_seq_len=None, runner_capabilities=None: [branch],
    )
    monkeypatch.setattr(
        search_mod, "resolve_backend_version", lambda hw, be: "1.3.0rc10"
    )


def _pc(*, tp=4, replicas=2):
    return ReplicaParallelConfig(
        ParallelShape(tp=tp, dp=1, moe_tp=1, moe_ep=tp), replicas=replicas
    )


def test_ranks_feasible_best_first_and_passes_replay_specs(monkeypatch):
    branch = _branch(_pc())  # 8 GPUs
    _stub(monkeypatch, branch)
    factory = _FakeRunnerFactory()

    candidates = _run_sweep(
        _config(),
        runner_factory=factory,
        sampler_factory=_FakeSampler,
        show_progress=False,
    )

    assert [candidate.score for candidate in candidates] == [768.0, 512.0, 256.0]
    assert all(candidate.used_gpus == 8 for candidate in candidates)
    assert all(
        candidate.config["backend_version"] == "1.3.0rc10" for candidate in candidates
    )
    assert candidates[0].metrics["gpu_hours"] == 1.0
    assert factory.worker_ids == [0]
    assert all(isinstance(spec, ReplaySpec) for spec in factory.runner.specs)
    assert factory.runner.closed


def test_parallel_batch_uses_worker_sized_timeout_waves(monkeypatch):
    branch = _branch(_pc())
    _stub(monkeypatch, branch)
    wave_sizes = []
    pools = []

    class ImmediateFuture:
        def __init__(self, result):
            self._result = result

        def result(self):
            return self._result

    class FakeProcessPool:
        def __init__(self, *, initializer, initargs, **kwargs):
            self._processes = {}
            self.shutdown_called = False
            self.shutdown_waits = []
            self.shutdown_wait = None
            initializer(*initargs)
            pools.append(self)

        def submit(self, function, *args):
            return ImmediateFuture(function(*args))

        def shutdown(self, *, wait, cancel_futures):
            self.shutdown_called = True
            self.shutdown_waits.append(wait)
            self.shutdown_wait = wait

    def fake_wait(pending, *, timeout, return_when):
        wave_sizes.append(len(pending))
        return set(pending), set()

    monkeypatch.setattr(search_mod, "ProcessPoolExecutor", FakeProcessPool)
    monkeypatch.setattr(search_mod, "wait", fake_wait)

    candidates = _run_sweep(
        _config(parallel_evals=2, max_eval_seconds=10),
        runner_factory=_FakeRunnerFactory(),
        sampler_factory=_FakeSampler,
        show_progress=False,
    )

    assert len(candidates) == 3
    assert wave_sizes == [2, 1]
    assert len(pools) == 1
    assert pools[0].shutdown_called
    assert pools[0].shutdown_waits == [True]
    assert pools[0].shutdown_wait is True


def test_timed_out_wave_is_gated_and_pool_is_replaced(monkeypatch):
    branch = _branch(_pc())
    _stub(monkeypatch, branch)
    pools = []
    sampler_seen = {}

    class PendingFuture:
        def result(self):
            raise AssertionError("a timed-out future must not be collected")

    class FakeProcessPool:
        def __init__(self, *, initializer, initargs, **kwargs):
            self._processes = {}
            self.shutdown_called = False
            self.shutdown_waits = []
            initializer(*initargs)
            pools.append(self)

        def submit(self, function, *args):
            return PendingFuture()

        def shutdown(self, *, wait, cancel_futures):
            self.shutdown_called = True
            self.shutdown_waits.append(wait)

    def factory(branch, study_id, objectives=None):
        sampler = _FakeSampler(branch, study_id, objectives)
        sampler_seen["sampler"] = sampler
        return sampler

    monkeypatch.setattr(search_mod, "ProcessPoolExecutor", FakeProcessPool)
    monkeypatch.setattr(
        search_mod, "wait", lambda pending, **kwargs: (set(), set(pending))
    )

    candidates = _run_sweep(
        _config(parallel_evals=2, max_eval_seconds=0.01),
        runner_factory=_FakeRunnerFactory(),
        sampler_factory=factory,
        show_progress=False,
    )

    assert candidates == []
    assert len(pools) > 1
    assert all(pool.shutdown_called for pool in pools)
    assert all(False in pool.shutdown_waits for pool in pools[:-1])
    assert pools[-1].shutdown_waits == [True]
    assert all(
        result[0] == "infeasible" and "exceed runtime" in result[1]
        for result in sampler_seen["sampler"].scored
    )


def test_broken_worker_pool_is_friendly_and_always_cleaned_up(monkeypatch):
    branch = _branch(_pc())
    _stub(monkeypatch, branch)
    pools = []

    class BrokenFuture:
        def result(self):
            raise search_mod.BrokenProcessPool("worker exited")

    class FakeProcessPool:
        def __init__(self, *, initializer, initargs, **kwargs):
            self._processes = {}
            self.shutdown_called = False
            self.shutdown_waits = []
            initializer(*initargs)
            pools.append(self)

        def submit(self, function, *args):
            return BrokenFuture()

        def shutdown(self, *, wait, cancel_futures):
            self.shutdown_called = True
            self.shutdown_waits.append(wait)

    monkeypatch.setattr(search_mod, "ProcessPoolExecutor", FakeProcessPool)
    monkeypatch.setattr(
        search_mod, "wait", lambda pending, **kwargs: (set(pending), set())
    )

    with pytest.raises(RuntimeError, match="guard a script entrypoint"):
        _run_sweep(
            _config(parallel_evals=2),
            runner_factory=_FakeRunnerFactory(),
            sampler_factory=_FakeSampler,
            show_progress=False,
        )

    assert pools[0].shutdown_called
    assert pools[0].shutdown_waits == [False]


def test_over_budget_candidates_are_observed_infeasible(monkeypatch):
    branch = _branch(_pc(tp=16, replicas=4))  # 64 GPUs
    _stub(monkeypatch, branch)
    sampler_seen = {}

    def factory(branch, study_id, objectives=None):
        sampler = _FakeSampler(branch, study_id, objectives)
        sampler_seen["sampler"] = sampler
        return sampler

    candidates = _run_sweep(
        _config(gpu_budget=32),
        runner_factory=_FakeRunnerFactory(),
        sampler_factory=factory,
        show_progress=False,
    )

    assert candidates == []
    scored = sampler_seen["sampler"].scored
    assert len(scored) == 33
    assert all(item[0] == "infeasible" for item in scored)
    assert all("over gpu_budget" in item[1] for item in scored)


def test_runner_failure_is_observed_infeasible(monkeypatch):
    branch = _branch(_pc())
    _stub(monkeypatch, branch)
    sampler_seen = {}

    class BoomRunner(_FakeRunner):
        def run(self, spec):
            raise RuntimeError("replay blew up")

    def factory(branch, study_id, objectives=None):
        sampler = _FakeSampler(branch, study_id, objectives)
        sampler_seen["sampler"] = sampler
        return sampler

    candidates = _run_sweep(
        _config(),
        runner_factory=_FakeRunnerFactory(BoomRunner()),
        sampler_factory=factory,
        show_progress=False,
    )

    assert candidates == []
    assert all(item[0] == "infeasible" for item in sampler_seen["sampler"].scored)


@pytest.mark.parametrize("invalid_metric", [float("nan"), float("inf"), True, "1"])
def test_runner_report_rejects_non_finite_or_non_numeric_metrics(invalid_metric):
    spec = ReplaySpec(
        backend_deployment=BackendDeploymentSpec(
            deployment_mode="agg",
            backend="vllm",
            backend_version="test",
        ),
        workload={},
        goal={},
    )

    class InvalidRunner:
        def run(self, replay_spec):
            assert replay_spec is spec
            return ReplayReport(metrics={"objective": invalid_metric})

    metrics, outcome, reason = search_mod._run_replay(spec, InvalidRunner())

    assert metrics is None
    assert outcome == "failed"
    assert "runner metric" in reason


def test_runner_report_requires_replay_report_and_strict_json_metadata():
    spec = ReplaySpec(
        backend_deployment=BackendDeploymentSpec(
            deployment_mode="agg",
            backend="vllm",
            backend_version="test",
        ),
        workload={},
        goal={},
    )

    class WrongTypeRunner:
        def run(self, replay_spec):
            return {"output_throughput_tok_s": 1.0}

    class InvalidMetadataRunner:
        def run(self, replay_spec):
            return ReplayReport(
                metrics={"output_throughput_tok_s": 1.0},
                metadata={"not_json": (1, 2)},
            )

    assert search_mod._run_replay(spec, WrongTypeRunner())[1] == "failed"
    result = search_mod._run_replay(spec, InvalidMetadataRunner())
    assert result[1] == "failed"
    assert "metadata" in result[2]


def test_goodput_goal_fails_closed_when_runner_omits_metric(monkeypatch):
    branch = _branch(_pc())
    _stub(monkeypatch, branch)
    sampler_seen = {}

    def factory(branch, study_id, objectives=None):
        sampler = _FakeSampler(branch, study_id, objectives)
        sampler_seen["sampler"] = sampler
        return sampler

    config = _config(candidates_per_round=1).model_copy(
        update={
            "goal": OptimizationGoal(
                target="goodput_per_gpu",
                sla={"ttft_ms": 2000.0, "itl_ms": 30.0},
            )
        }
    )
    candidates = _run_sweep(
        config,
        runner_factory=_FakeRunnerFactory(),
        sampler_factory=factory,
        show_progress=False,
    )

    assert candidates == []
    assert sampler_seen["sampler"].scored
    assert all(
        item[0] == "infeasible" and "goodput_output_throughput_tok_s" in item[1]
        for item in sampler_seen["sampler"].scored
    )


def test_replay_spec_version_is_checked_before_runner_creation(monkeypatch):
    branch = _branch(_pc())
    _stub(monkeypatch, branch)

    class IncompatibleFactory(_FakeRunnerFactory):
        def capabilities(self):
            return RunnerCapabilities(
                replay_spec_api_version=2,
                supported_backend_topologies=(("*", "*"),),
            )

    factory = IncompatibleFactory()
    with pytest.raises(ValueError, match="runner version 2"):
        _run_sweep(
            _config(),
            runner_factory=factory,
            sampler_factory=_FakeSampler,
            show_progress=False,
        )

    assert factory.worker_ids == []


def test_unsupported_backend_pair_never_reaches_runner(monkeypatch):
    pc = _pc()
    branch = BranchSpace(
        deployment_mode="agg",
        parallel_configs=(pc,),
        supported_backends={pc: frozenset({"vllm"})},
        knob_choices={"backend": ["vllm", "trtllm"]},
    )
    _stub(monkeypatch, branch)
    sampler_seen = {}

    class NeverCalledRunner(_FakeRunner):
        def run(self, spec):
            raise AssertionError("unsupported pairs must not reach replay")

    def factory(branch, study_id, objectives=None):
        sampler = _FakeSampler(branch, study_id, objectives)
        sampler_seen["sampler"] = sampler
        return sampler

    candidates = _run_sweep(
        _config(),
        runner_factory=_FakeRunnerFactory(NeverCalledRunner()),
        sampler_factory=factory,
        show_progress=False,
    )

    assert candidates == []
    scored = sampler_seen["sampler"].scored
    assert len(scored) == 33
    assert all(item[0] == "infeasible" for item in scored)
    assert all("does not support" in item[1] for item in scored)


def test_study_id_is_unique_per_run(monkeypatch):
    branch = _branch(_pc())
    _stub(monkeypatch, branch)
    seen = []

    def factory(branch, study_id, objectives=None):
        seen.append(study_id)
        return _FakeSampler(branch, study_id, objectives)

    sweeper = Sweeper(
        runner_factory=_FakeRunnerFactory(),
        sampler_factory=factory,
        show_progress=False,
    )
    for _ in range(2):
        sweeper.run(_config())

    assert len(seen) == 2 and seen[0] != seen[1]
    assert all(study_id.startswith("sweeper_agg_") for study_id in seen)


def test_candidate_build_error_is_reported_not_raised(monkeypatch, capsys):
    branch = _branch(_pc())
    _stub(monkeypatch, branch)
    sampler_seen = {}

    class BadSampler(_FakeSampler):
        def suggest(self, count):
            selection = {
                "deployment_mode": "agg",
                "backend": "trtllm",
                "agg_max_num_batched_tokens": 8192,
            }
            return [
                Suggestion(
                    selection=selection,
                    parallel_config=self.branch.parallel_configs[0],
                    handle=selection,
                )
                for _ in range(count)
            ]

    def factory(branch, study_id, objectives=None):
        sampler = BadSampler(branch, study_id, objectives)
        sampler_seen["sampler"] = sampler
        return sampler

    candidates = _run_sweep(
        _config(),
        runner_factory=_FakeRunnerFactory(),
        sampler_factory=factory,
        show_progress=True,
    )

    assert candidates == []
    scored = sampler_seen["sampler"].scored
    assert len(scored) == 33
    assert scored[0][0] == "infeasible"
    assert "candidate build failed" in scored[0][1]
    output = capsys.readouterr().out
    assert "Sweeper failure reason(s): candidate build failed" in output
    assert "(x33)" in output


def test_duplicate_full_samples_use_cache_and_are_replaced(monkeypatch):
    branch = _branch(_pc())
    _stub(monkeypatch, branch)
    seen = {}

    class DuplicateThenUniqueSampler(_FakeSampler):
        def __init__(self, branch, study_id, objectives=None):
            super().__init__(branch, study_id, objectives)
            self.ask_no = 0

        def suggest(self, count):
            self.ask_no += 1
            seqs = [256] * count if self.ask_no == 1 else [512, 768][:count]
            return [
                Suggestion(
                    selection=(selection := _selection(value)),
                    parallel_config=self.branch.parallel_configs[0],
                    handle=selection,
                )
                for value in seqs
            ]

    def factory(branch, study_id, objectives=None):
        sampler = DuplicateThenUniqueSampler(branch, study_id, objectives)
        seen["sampler"] = sampler
        return sampler

    runner = _FakeRunner()
    candidates = _run_sweep(
        _config(),
        runner_factory=_FakeRunnerFactory(runner),
        sampler_factory=factory,
        show_progress=False,
    )

    assert runner.calls == 3
    assert {candidate.config["agg_max_num_seqs"] for candidate in candidates} == {
        256,
        512,
        768,
    }
    assert len(seen["sampler"].scored) == 5


def test_cache_identity_preserves_distinct_json_scalar_types():
    parallel = _pc()
    boolean = Suggestion(
        selection={"adapter::example::choice": True},
        parallel_config=parallel,
        handle=None,
    )
    integer = Suggestion(
        selection={"adapter::example::choice": 1},
        parallel_config=parallel,
        handle=None,
    )
    floating = Suggestion(
        selection={"adapter::example::choice": 1.0},
        parallel_config=parallel,
        handle=None,
    )

    keys = {
        search_mod._suggestion_cache_key(suggestion, ("same-context",))
        for suggestion in (boolean, integer, floating)
    }
    assert len(keys) == 3


def test_projection_stall_only_stops_current_branch(monkeypatch):
    parallel = _pc()
    agg = _branch(parallel)
    disagg = BranchSpace(
        deployment_mode="disagg",
        parallel_configs=(parallel,),
        supported_backends={parallel: frozenset({"trtllm"})},
        knob_choices={"backend": ["trtllm"]},
    )
    monkeypatch.setattr(
        search_mod,
        "enumerate_branches",
        lambda config, *, max_seq_len=None, runner_capabilities=None: [agg, disagg],
    )
    monkeypatch.setattr(
        search_mod, "resolve_backend_version", lambda hw, be: "1.3.0rc10"
    )
    seen = []

    class RepeatingSampler(_FakeSampler):
        def suggest(self, count):
            suggestions = super().suggest(1)
            return suggestions * count

    class EmptySampler(_FakeSampler):
        def suggest(self, count):
            return []

    def factory(branch, study_id, objectives=None):
        seen.append(branch.deployment_mode)
        sampler_type = (
            RepeatingSampler if branch.deployment_mode == "agg" else EmptySampler
        )
        return sampler_type(branch, study_id, objectives)

    _run_sweep(
        _config(),
        runner_factory=_FakeRunnerFactory(),
        sampler_factory=factory,
        show_progress=False,
    )

    assert seen == ["agg", "disagg"]


def _pareto_config():
    return SmartSearchConfig(
        search_space={
            "model_name": "deepseek-ai/DeepSeek-V3",
            "hardware_sku": "gb200",
            "backend": ["trtllm"],
            "deployment_mode": ["agg"],
            "gpu_budget": 32,
        },
        workload={
            "isl": 1024,
            "osl": 1024,
            "kv_load_ratio": [0.0, 1.0],
            "num_request_ratio": 10,
        },
        sweep={"max_rounds": 1, "candidates_per_round": 3, "parallel_evals": 1},
        goal={"target": "pareto"},
    )


_PARETO_POINTS = {4: (100.0, 40.0), 8: (150.0, 25.0), 16: (180.0, 12.0)}


class _ParetoSampler:
    def __init__(self, branch, study_id, objectives=None):
        self.branch = branch
        self.objectives = objectives
        self.observed: list = []

    def suggest(self, count):
        return [
            Suggestion(
                selection={
                    **_selection(256),
                    "kv_load_ratio": ratio,
                },
                parallel_config=self.branch.parallel_configs[0],
                handle={"kv_load_ratio": ratio},
            )
            for ratio in (0.25, 0.5, 1.0)
        ]

    def observe(self, suggestion, metrics):
        self.observed.append(metrics)

    def observe_infeasible(self, suggestion, reason):
        self.observed.append(("infeasible", reason))


class _ParetoRunner(_FakeRunner):
    def run(self, spec):
        assert spec.concurrency is not None
        throughput, per_user = _PARETO_POINTS[spec.concurrency]
        return ReplayReport(
            metrics={
                "output_throughput_tok_s": throughput,
                "mean_output_token_throughput_per_user": per_user,
                "gpu_hours": 1.0,
                "duration_ms": 3_600_000.0,
            }
        )


def test_pareto_sweep_preserves_kv_load_and_returns_front(monkeypatch):
    branch = _branch(_pc())
    _stub(monkeypatch, branch)
    concurrency_by_ratio = {0.25: 4, 0.5: 8, 1.0: 16}

    def fake_resolve(sample, *, workload, parallel_config, ratio, backend_version):
        return KVLoadResolution(
            ratio=ratio,
            concurrency=concurrency_by_ratio[ratio],
            concurrency_capacity=16,
            role_capacity_tokens={"agg": 24_576},
        )

    monkeypatch.setattr(search_mod, "resolve_kv_load", fake_resolve)
    seen = {}

    def factory(branch, study_id, objectives=None):
        sampler = _ParetoSampler(branch, study_id, objectives)
        seen["sampler"] = sampler
        return sampler

    front = _run_sweep(
        _pareto_config(),
        runner_factory=_FakeRunnerFactory(_ParetoRunner()),
        sampler_factory=factory,
        show_progress=False,
    )

    assert [c.objectives["throughput_per_user"] for c in front] == [12.0, 25.0, 40.0]
    assert [c.objectives["throughput_per_gpu"] for c in front] == [
        180.0,
        150.0,
        100.0,
    ]
    assert {candidate.config["concurrency"] for candidate in front} == {4, 8, 16}
    assert {candidate.config["kv_load_ratio"] for candidate in front} == {
        0.25,
        0.5,
        1.0,
    }
    assert seen["sampler"].objectives == [
        ("throughput_per_gpu", True),
        ("throughput_per_user", True),
    ]
    assert all(
        set(metrics) == {"throughput_per_gpu", "throughput_per_user"}
        for metrics in seen["sampler"].observed
    )
