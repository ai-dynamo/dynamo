# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit coverage for the Dynamo-neutral adapter and replay contracts."""

from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass
from enum import Enum

import pytest

from aisimulate import sweeper
from aisimulate.sweeper.provider import (
    API_VERSION,
    AdapterReplaySpec,
    AdapterSearchPlan,
    CandidateContext,
    RuntimeHookSpec,
    SearchSpaceFragment,
    SweepConfigProvider,
    SweepContext,
)
from aisimulate.sweeper.replay import (
    REPLAY_SPEC_API_VERSION,
    BackendDeploymentSpec,
    HookCapability,
    ReplayOutputRequirements,
    ReplayReport,
    ReplaySpec,
    RunnerCapabilities,
    canonical_json,
    validate_json_value,
)


def _deployment(*, mode: str = "agg", backend: str = "vllm"):
    return BackendDeploymentSpec(
        deployment_mode=mode,
        backend=backend,
        backend_version="0.1",
        parallel_config={"tp": 2, "replicas": 1},
        agg_engine_args={"max_num_seqs": 256},
        num_workers=1,
    )


def _planner_hook(*, version: int = 1):
    return RuntimeHookSpec(
        provider="dynamo.planner",
        kind="scaling",
        api_version=version,
        config={"interval_seconds": 180},
    )


def _replay_spec(*, hook: RuntimeHookSpec | None = None):
    adapter = (
        {}
        if hook is None
        else {
            "dynamo.planner": AdapterReplaySpec(
                config={"enabled": True}, runtime_hooks=(hook,)
            )
        }
    )
    return ReplaySpec(
        backend_deployment=_deployment(),
        workload={"isl": 128, "osl": 32, "request_rate": 2.0},
        goal={"target": "throughput"},
        concurrency=4,
        adapters=adapter,
    )


def test_contracts_pickle_and_canonical_json_round_trip():
    hook = _planner_hook()
    fragment = SearchSpaceFragment(
        choices_by_branch={"agg": {"scaling_policy": ["disabled", "load"]}},
        float_ranges_by_branch={"agg": {"sensitivity": (0.0, 1.0)}},
    )
    search_plan = AdapterSearchPlan(
        fragment=fragment,
        state={"winner_by_interval": {"180": "constant"}},
        diagnostics={"loss": 0.25},
        potential_runtime_hooks=(hook,),
    )
    deployment = _deployment()
    values = [
        SweepContext(
            core_search_space={"backend": ["vllm"]},
            workload={"request_rate": 2.0},
            goal={"target": "throughput"},
            show_progress=False,
        ),
        CandidateContext(
            sample={"backend": "vllm"},
            backend_deployment=deployment,
            concurrency=4,
        ),
        fragment,
        search_plan,
        AdapterReplaySpec(config={"enabled": True}, runtime_hooks=(hook,)),
        deployment,
        _replay_spec(hook=hook),
        ReplayReport(metrics={"output_throughput_tok_s": 10.0}),
        ReplayOutputRequirements(
            include_raw_report=True,
            capture_per_request=True,
        ),
        RunnerCapabilities(
            supported_backend_topologies=(("vllm", "agg"),),
            supported_hooks=(HookCapability("dynamo.planner", "scaling", 1),),
        ),
    ]

    for value in values:
        assert pickle.loads(pickle.dumps(value)) == value
        assert json.loads(canonical_json(value)) is not None


def test_canonical_json_is_stable_and_strict():
    left = {"z": [2, 1], "a": {"second": 2, "first": 1}}
    right = {"a": {"first": 1, "second": 2}, "z": [2, 1]}

    assert canonical_json(left) == canonical_json(right)
    assert canonical_json(left) == '{"a":{"first":1,"second":2},"z":[2,1]}'
    with pytest.raises(ValueError, match="Out of range float values"):
        canonical_json({"metric": math.nan})
    with pytest.raises(TypeError, match="string mapping keys"):
        canonical_json({1: "not allowed"})
    with pytest.raises(TypeError, match="not supported"):
        canonical_json(object())


def test_adapter_payload_json_validation_does_not_normalize_python_objects():
    @dataclass
    class PythonObject:
        value: int

    class StringEnum(str, Enum):
        VALUE = "value"

    validate_json_value({"nested": [None, True, 1, 1.0, "value"]})
    for value in (PythonObject(1), StringEnum.VALUE, (1, 2)):
        with pytest.raises(TypeError, match="non-JSON value"):
            validate_json_value({"nested": value})
    with pytest.raises(ValueError, match="finite JSON numbers"):
        validate_json_value({"nested": math.inf})


def test_replay_spec_collects_runtime_hooks_in_adapter_order():
    planner = _planner_hook()
    router = RuntimeHookSpec("dynamo.router", "placement", 2, {})
    spec = ReplaySpec(
        backend_deployment=_deployment(),
        workload={},
        goal={},
        adapters={
            "dynamo.router": AdapterReplaySpec(runtime_hooks=(router,)),
            "dynamo.planner": AdapterReplaySpec(runtime_hooks=(planner,)),
        },
    )

    assert spec.runtime_hooks == (router, planner)


def test_runner_capabilities_accept_supported_spec_and_wildcards():
    hook = _planner_hook()
    capabilities = RunnerCapabilities(
        supported_backend_topologies=(("vllm", "*"),),
        supported_hooks=(HookCapability("dynamo.planner", "scaling", 1),),
    )

    assert capabilities.supports_backend_topology("vllm", "agg")
    assert not capabilities.supports_backend_topology("sglang", "agg")
    assert capabilities.supports_hook(hook)
    capabilities.require_compatible(_replay_spec(hook=hook))


def test_runner_capabilities_reject_spec_version_backend_and_hook():
    capabilities = RunnerCapabilities(
        supported_backend_topologies=(("vllm", "agg"),),
        supported_hooks=(HookCapability("dynamo.planner", "scaling", 1),),
    )
    wrong_version = ReplaySpec(
        backend_deployment=_deployment(),
        workload={},
        goal={},
        api_version=REPLAY_SPEC_API_VERSION + 1,
    )
    with pytest.raises(ValueError, match="ReplaySpec API version"):
        capabilities.require_compatible(wrong_version)
    with pytest.raises(ValueError, match="runner version 1"):
        capabilities.require_replay_spec_version(REPLAY_SPEC_API_VERSION + 1)

    wrong_backend = ReplaySpec(
        backend_deployment=_deployment(backend="sglang"), workload={}, goal={}
    )
    with pytest.raises(ValueError, match="sglang.*agg"):
        capabilities.require_compatible(wrong_backend)

    with pytest.raises(ValueError, match=r"dynamo\.planner:scaling@2"):
        capabilities.require_compatible(_replay_spec(hook=_planner_hook(version=2)))


def test_sweep_config_provider_protocol_is_structural():
    class Adapter:
        name = "example"
        api_version = API_VERSION

        def generate_search_space(self, search_spec, context):
            return AdapterSearchPlan()

        def materialize_replay(self, plan, selection, context):
            return AdapterReplaySpec()

    assert isinstance(Adapter(), SweepConfigProvider)


def test_public_contract_versions_start_at_one():
    assert API_VERSION == 1
    assert REPLAY_SPEC_API_VERSION == 1


def test_lazy_exports_are_listed_in_public_api():
    assert set(sweeper._LAZY_EXPORTS).issubset(sweeper.__all__)
