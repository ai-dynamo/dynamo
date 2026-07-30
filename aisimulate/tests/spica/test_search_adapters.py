# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for adapter search-space and ReplaySpec materialization."""

from pathlib import Path

import pytest

import aisimulate.spica.search as search_module
from aisimulate.spica.adapter import (
    AdapterReplaySpec,
    AdapterSearchPlan,
    RuntimeHookSpec,
    SearchSpaceFragment,
)
from aisimulate.spica.config import SmartSearchConfig
from aisimulate.spica.parallel_enum import ParallelShape, ReplicaParallelConfig
from aisimulate.spica.replay import HookCapability, ReplayReport, RunnerCapabilities
from aisimulate.spica.sampler import Suggestion
from aisimulate.spica.search_space import BranchSpace

TRACE = str(Path(__file__).parent / "data" / "mooncake_tiny.jsonl")
_HOOK = RuntimeHookSpec(
    provider="test.feature",
    kind="policy",
    api_version=1,
)


def _config() -> SmartSearchConfig:
    return SmartSearchConfig(
        search_space={
            "model_name": "model",
            "hardware_sku": "h200_sxm",
            "backend": ["vllm"],
            "deployment_mode": ["agg"],
        },
        adapters={
            "test.feature": {
                "search_space": {"modes": ["fast"]},
            }
        },
        workload={"trace_path": TRACE},
        sweep={"max_rounds": 1, "candidates_per_round": 1, "parallel_evals": 1},
    )


class _Adapter:
    name = "test.feature"
    api_version = 1

    def __init__(self) -> None:
        self.generated = []
        self.materialized = []

    def generate_search_space(self, search_spec, context):
        self.generated.append((search_spec, context))
        return AdapterSearchPlan(
            fragment=SearchSpaceFragment(
                choices_by_branch={"agg": {"mode": list(search_spec["modes"])}}
            ),
            potential_runtime_hooks=(_HOOK,),
        )

    def materialize_replay(self, plan, selection, context):
        self.materialized.append((plan, selection, context))
        hook = RuntimeHookSpec(
            provider=_HOOK.provider,
            kind=_HOOK.kind,
            api_version=_HOOK.api_version,
            config={"mode": selection["mode"]},
        )
        return AdapterReplaySpec(
            config={"mode": selection["mode"]},
            runtime_hooks=(hook,),
        )


class _Sampler:
    def __init__(self, branch, study_id, objectives=None):
        del study_id, objectives
        self.branch = branch
        assert branch.knob_choices["adapter::test.feature::mode"] == ["fast"]

    def suggest(self, count):
        assert count == 1
        selection = {
            "deployment_mode": "agg",
            "backend": "vllm",
            "agg_max_num_batched_tokens": 8192,
            "agg_max_num_seqs": 256,
            "adapter::test.feature::mode": "fast",
        }
        return [
            Suggestion(
                selection=selection,
                parallel_config=self.branch.parallel_configs[0],
                handle=selection,
            )
        ]

    def observe(self, suggestion, metrics):
        del suggestion, metrics

    def observe_infeasible(self, suggestion, reason):
        pytest.fail(f"unexpected infeasible suggestion {suggestion}: {reason}")


class _Runner:
    def __init__(self) -> None:
        self.specs = []
        self.closed = False

    def run(self, spec):
        self.specs.append(spec)
        return ReplayReport(metrics={"output_throughput_tok_s": 12.0})

    def close(self):
        self.closed = True


class _RunnerFactory:
    def __init__(self, *, support_hook: bool = True) -> None:
        self.runner = _Runner()
        self.support_hook = support_hook
        self.created = 0

    def capabilities(self):
        hooks = (
            (HookCapability("test.feature", "policy", 1),) if self.support_hook else ()
        )
        return RunnerCapabilities(
            supported_backend_topologies=(("*", "*"),),
            supported_hooks=hooks,
        )

    def create(self, worker_id):
        assert worker_id == 0
        self.created += 1
        return self.runner


def _stub_branch(monkeypatch) -> None:
    parallel = ReplicaParallelConfig(
        shape=ParallelShape(tp=1, dp=1, moe_tp=1, moe_ep=1),
        replicas=1,
    )
    branch = BranchSpace(
        deployment_mode="agg",
        parallel_configs=(parallel,),
        supported_backends={parallel: frozenset({"vllm"})},
        knob_choices={
            "backend": ["vllm"],
            "agg_max_num_batched_tokens": [8192],
            "agg_max_num_seqs": [256],
        },
    )
    monkeypatch.setattr(
        search_module,
        "enumerate_branches",
        lambda config, *, max_seq_len=None, runner_capabilities=None: [branch],
    )
    monkeypatch.setattr(
        search_module,
        "resolve_backend_version",
        lambda hardware, backend: "0.11.0",
    )


def test_adapter_accepts_search_space_and_materializes_spec_on_main(
    monkeypatch,
) -> None:
    _stub_branch(monkeypatch)
    adapter = _Adapter()
    factory = _RunnerFactory()

    candidates = search_module.run_smart_search(
        _config(),
        runner_factory=factory,
        adapters={"test.feature": adapter},
        sampler_factory=_Sampler,
        show_progress=False,
    )

    assert adapter.generated[0][0] == {"modes": ["fast"]}
    assert adapter.materialized[0][1] == {"mode": "fast"}
    assert factory.created == 1
    assert factory.runner.closed
    spec = factory.runner.specs[0]
    assert spec.adapters["test.feature"].config == {"mode": "fast"}
    assert spec.runtime_hooks[0].config == {"mode": "fast"}
    assert candidates[0].config["adapters"] == {"test.feature": {"mode": "fast"}}


def test_runner_hook_capability_is_checked_before_runner_creation(monkeypatch) -> None:
    _stub_branch(monkeypatch)
    factory = _RunnerFactory(support_hook=False)

    with pytest.raises(ValueError, match="unsupported runtime hook"):
        search_module.run_smart_search(
            _config(),
            runner_factory=factory,
            adapters={"test.feature": _Adapter()},
            sampler_factory=_Sampler,
            show_progress=False,
        )

    assert factory.created == 0
