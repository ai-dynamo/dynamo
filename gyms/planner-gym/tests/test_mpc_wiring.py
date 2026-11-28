# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Match Config and factory wiring of MPC.

Config resolution needs no Dynamo runtime; the tests that build real adapters
skip without it. Controller behaviour is covered by ``test_mpc.py``.
"""

from __future__ import annotations

import asyncio
import sys
import textwrap
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from autoscaling_arena import match_runner
from autoscaling_arena.match_config import (
    MatchConfigError,
    SimAutoscalerConfig,
    SimBackendConfig,
    feasible_worker_targets,
    load_match_config,
)

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]

ARENA_ROOT = Path(__file__).resolve().parents[1]
PLANNER_CONFIG = ARENA_ROOT / "configs" / "planner.sim.example.yaml"
MPC_EXAMPLE = ARENA_ROOT / "configs" / "match.mpc.example.yaml"
COGNITIVE_AGENT_EXAMPLE = ARENA_ROOT / "configs" / "match.cognitive-agent.example.yaml"

# ``substrate: gpt_oss`` resolves a separate AIS identity for each deployed role.
EXPECTED_AIS_SPEC = {
    "type": "ais",
    "config": {
        "model": "openai/gpt-oss-120b",
        "system": "h200_sxm",
        "backend": "vllm",
        "backend_version": "0.19.0",
        "tp": 1,
        "moe_tp_size": 1,
        "moe_ep_size": 1,
        "attention_dp": 1,
        "worker_type": "prefill",
        "estimation_mode": "auto",
        "fallback_policy": "deny",
    },
}
EXPECTED_AIS_SPEC["decode_config"] = {
    **EXPECTED_AIS_SPEC["config"],
    "worker_type": "decode",
}


def _write_sim_config(
    tmp_path: Path,
    *,
    autoscalers: str,
    topology: str = "disagg",
    gpu_budget: int = 32,
    backend_extra: str = "",
    substrate: bool = True,
) -> Path:
    entries = textwrap.indent(textwrap.dedent(autoscalers).strip(), "    ")
    extra = textwrap.indent(textwrap.dedent(backend_extra).strip(), "  ")
    body = textwrap.dedent(
        f"""
        schema_version: 1
        name: mpc-contract
        backend:
          type: sim
          substrate: gpt_oss
          topology: {topology}
          gpu_budget: {gpu_budget}
          planner_config: {PLANNER_CONFIG}
        @@BACKEND_EXTRA@@
          autoscalers:
        @@AUTOSCALERS@@
        evaluations:
          workloads: [flat]
          defaults:
            seed: 0
            max_requests: 50
            arrival_speedup: 1.0
        slo_profiles:
          - name: interactive
            ttft_ms: 300
            itl_ms: 50
        metrics:
          rank_by: goodput_per_gpu
          include: [goodput_per_gpu]
        execution:
          repetitions: 1
          fail_fast: false
        publish:
          artifact_root: out
          destinations:
            - type: console
        """
    ).lstrip()
    path = tmp_path / "match.yaml"
    if not substrate:
        # Explicit model and engines (in ``backend_extra``) replace the preset.
        body = body.replace("  substrate: gpt_oss\n", "")
    path.write_text(
        body.replace("@@BACKEND_EXTRA@@", extra).replace("@@AUTOSCALERS@@", entries)
    )
    return path


def _sim_autoscalers(path: Path) -> tuple[SimAutoscalerConfig, ...]:
    backend = load_match_config(path).backend
    assert isinstance(backend, SimBackendConfig)
    return backend.autoscalers


# --------------------------------------------------------------------------- #
# Match Config
# --------------------------------------------------------------------------- #


def test_mpc_defaults_resolve_forward_model_from_backend(tmp_path: Path):
    (mpc,) = _sim_autoscalers(
        _write_sim_config(
            tmp_path,
            autoscalers="""
            - name: mpc
              type: mpc
              start: {prefill: 1, decode: 1}
            """,
        )
    )
    assert mpc.type == "mpc"
    assert mpc.config["forward_model"] == EXPECTED_AIS_SPEC
    assert mpc.config["gpus_per_worker"] == 1
    assert mpc.config["prefill_gpus_per_worker"] == 1  # disagg: prefill pool cost
    assert mpc.config["max_replicas"] == 8 and mpc.config["poll_interval_s"] == 5.0
    # Startup delay follows the deployed decode engine unless set explicitly.
    assert mpc.config["cold_start_s"] == 60.0
    assert mpc.start.prefill == 1 and mpc.start.decode == 1


def test_mpc_roofline_in_agg_topology_resolves_without_engine_spec(
    tmp_path: Path,
):
    (mpc,) = _sim_autoscalers(
        _write_sim_config(
            tmp_path,
            topology="agg",
            autoscalers="""
            - name: mpc-lite
              type: mpc
              start: {decode: 1}
              config: {forward_model: roofline}
            """,
        )
    )
    assert mpc.config["forward_model"] == {"type": "roofline"}
    assert "prefill_gpus_per_worker" not in mpc.config


@pytest.mark.parametrize(
    ("autoscaler", "message_parts"),
    [
        (
            """
            - name: bad
              type: mpc
              start: {prefill: 1, decode: 1}
              config: {min_prefill: 1}
            """,
            ("unknown fields", "min_prefill"),
        ),
        (
            """
            - name: bad
              type: mpc
              start: {prefill: 1, decode: 1}
              config: {smoothing_alpha: 1.5}
            """,
            ("smoothing_alpha", "unknown fields"),
        ),
        (
            """
            - name: bad
              type: mpc
              start: {prefill: 1, decode: 1}
              config: {min_replicas: 5, max_replicas: 2}
            """,
            ("min_replicas must be <= max_replicas",),
        ),
        (
            """
            - name: bad
              type: mpc
              start: {prefill: 1, decode: 9}
            """,
            ("start.decode", "[1, 8]"),
        ),
        (
            # Replay starts workers after the engine's delay, here 60 seconds.
            """
            - name: bad
              type: mpc
              start: {prefill: 1, decode: 1}
              config: {cold_start_s: 10}
            """,
            ("cold_start_s", "runtime.cold_start_delay_s (60 s)"),
        ),
        (
            # The controller's targets must be the ones the run is scored on.
            """
            - name: bad
              type: mpc
              start: {prefill: 1, decode: 1}
              config: {slo_ttft_ms: 2000}
            """,
            ("slo_ttft_ms and slo_itl_ms must equal SLO profile", "interactive"),
        ),
        (
            # Scoring and replay must charge a worker the same GPUs.
            """
            - name: bad
              type: mpc
              start: {prefill: 1, decode: 1}
              config: {gpus_per_worker: 4}
            """,
            ("gpus_per_worker", "deployed engine's 1 GPUs per worker"),
        ),
        (
            """
            - name: bad
              type: mpc
              start: {prefill: 1, decode: 1}
              config: {prefill_gpus_per_worker: 2}
            """,
            ("prefill_gpus_per_worker", "deployed engine's 1 GPUs per worker"),
        ),
        (
            # A prediction made between samples never matches a telemetry interval.
            """
            - name: bad
              type: mpc
              start: {prefill: 1, decode: 1}
              config: {poll_interval_s: 2}
            """,
            ("poll_interval_s", "multiple of the 5-second telemetry interval"),
        ),
        (
            """
            - name: bad
              type: mpc
              start: {prefill: 1, decode: 1}
              config: {poll_interval_s: 7.5}
            """,
            ("poll_interval_s", "multiple of the 5-second telemetry interval"),
        ),
    ],
)
def test_mpc_config_errors(tmp_path: Path, autoscaler: str, message_parts):
    with pytest.raises(MatchConfigError) as excinfo:
        load_match_config(_write_sim_config(tmp_path, autoscalers=autoscaler))
    for part in message_parts:
        assert part in str(excinfo.value)


MPC_ENTRY = """
- name: mpc
  type: mpc
  start: {prefill: 1, decode: 1}
"""


def test_mpc_accepts_matching_costs_and_aligned_cadence(tmp_path: Path):
    (mpc,) = _sim_autoscalers(
        _write_sim_config(
            tmp_path,
            autoscalers="""
            - name: mpc
              type: mpc
              start: {prefill: 1, decode: 1}
              config:
                gpus_per_worker: 1
                prefill_gpus_per_worker: 1
                cold_start_s: 60
                poll_interval_s: 10
            """,
        )
    )
    assert mpc.config["poll_interval_s"] == 10
    assert mpc.config["cold_start_s"] == 60


def _asymmetric_mpc_config(tmp_path: Path, start: str) -> Path:
    """Four-GPU prefill and eight-GPU decode workers under an 80-GPU budget."""
    engines = {
        "common": {"system": "h200_sxm", "backend": "vllm"},
        "prefill": {"tp_size": 4},
        "decode": {"tp_size": 8},
    }
    model = {"name": "served-model", "ais_model_path": "perf-db/model"}
    return _write_sim_config(
        tmp_path,
        autoscalers=f"""
        - name: mpc
          type: mpc
          start: {start}
          config: {{max_replicas: 15}}
        """,
        gpu_budget=80,
        substrate=False,
        backend_extra=yaml.safe_dump({"model": model, "engines": engines}),
    )


def test_disaggregated_mpc_searches_asymmetric_fleets_within_the_budget(
    tmp_path: Path,
):
    backend = load_match_config(
        _asymmetric_mpc_config(tmp_path, "{prefill: 1, decode: 1}")
    ).backend
    (mpc,) = backend.autoscalers
    targets = feasible_worker_targets(backend, mpc)
    # Both pools at 15 workers would need 180 GPUs. The controller chooses
    # only among the pairs that fit, so the wide prefill pool stays reachable.
    assert (15, 1) in targets and (2, 9) in targets
    assert all(4 * prefill + 8 * decode <= 80 for prefill, decode in targets)

    with pytest.raises(MatchConfigError, match="starting fleet requires 84 GPUs"):
        load_match_config(_asymmetric_mpc_config(tmp_path, "{prefill: 15, decode: 3}"))


def test_aggregated_mpc_ceiling_must_fit_the_gpu_budget(tmp_path: Path):
    with pytest.raises(MatchConfigError) as excinfo:
        load_match_config(
            _write_sim_config(
                tmp_path,
                topology="agg",
                autoscalers="""
                - name: bad
                  type: mpc
                  start: {decode: 1}
                  config: {max_replicas: 40}
                """,
            )
        )
    for part in ("maximum fleet requires 40 GPUs", "gpu_budget=32", "max_replicas"):
        assert part in str(excinfo.value)


@pytest.mark.parametrize("interval", ["null", "10"])
def test_mpc_requires_five_second_telemetry(tmp_path: Path, interval: str):
    with pytest.raises(MatchConfigError, match="5-second replay telemetry"):
        load_match_config(
            _write_sim_config(
                tmp_path,
                autoscalers=MPC_ENTRY,
                backend_extra=f"replay: {{telemetry_sample_interval_s: {interval}}}",
            )
        )


def test_mpc_requires_one_cold_start_delay_for_both_pools(tmp_path: Path):
    example = yaml.safe_load(COGNITIVE_AGENT_EXAMPLE.read_text())["backend"]

    def explicit_engines(prefill: float, decode: float) -> str:
        engines = {
            "common": example["engines"]["common"],
            "prefill": {"runtime": {"cold_start_delay_s": prefill}},
            "decode": {"runtime": {"cold_start_delay_s": decode}},
        }
        return yaml.safe_dump({"model": example["model"], "engines": engines})

    def load(prefill: float, decode: float):
        return _sim_autoscalers(
            _write_sim_config(
                tmp_path,
                autoscalers=MPC_ENTRY,
                substrate=False,
                backend_extra=explicit_engines(prefill, decode),
            )
        )

    with pytest.raises(MatchConfigError, match="single cold-start delay"):
        load(60, 10)
    # A null delay hands the startup time to replay's default, unseen here.
    with pytest.raises(MatchConfigError, match="cold_start_delay_s to a number"):
        load(None, None)
    (mpc,) = load(45, 45)
    assert mpc.config["cold_start_s"] == 45


def test_unknown_autoscaler_type_lists_supported_choices(tmp_path: Path):
    with pytest.raises(MatchConfigError) as excinfo:
        load_match_config(
            _write_sim_config(
                tmp_path,
                autoscalers="""
                - name: bad
                  type: hpa
                  start: {prefill: 1, decode: 1}
                """,
            )
        )
    message = str(excinfo.value)
    assert all(
        name in message
        for name in ("mpc", "planner", "keda", "reactive", "static", "jev")
    )


def test_mpc_example_expands_the_roster():
    config = load_match_config(MPC_EXAMPLE)
    backend = config.backend
    assert isinstance(backend, SimBackendConfig)
    types_by_name = {entry.name: entry.type for entry in backend.autoscalers}
    assert types_by_name["cloudAI-MPC"] == "mpc"
    assert set(types_by_name.values()) == {
        "planner",
        "keda",
        "reactive",
        "static",
        "mpc",
    }
    assert backend.topology == "agg" and backend.replay.telemetry_sample_interval_s == 5
    assert config.expected_runs == len(list(config.iter_runs())) == 10


# --------------------------------------------------------------------------- #
# Sim factory (lazy adapter boundary replaced, like the static-factory test)
# --------------------------------------------------------------------------- #


def _install_fake_mpc_module(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    created: dict[str, Any] = {"ais": [], "roofline": []}

    class FakeAISCapacityModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            created["ais"].append(self)

    class FakeRooflineCapacityModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            created["roofline"].append(self)

    class FakeMPC:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_module = types.ModuleType("autoscaling_arena.adapters.mpc")
    fake_module.AISCapacityModel = FakeAISCapacityModel  # type: ignore[attr-defined]
    fake_module.RooflineCapacityModel = FakeRooflineCapacityModel  # type: ignore[attr-defined]
    fake_module.MPCAutoscaler = FakeMPC  # type: ignore[attr-defined]
    fake_adapters = types.ModuleType("autoscaling_arena.adapters")
    fake_adapters.mpc = fake_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "autoscaling_arena.adapters", fake_adapters)
    monkeypatch.setitem(sys.modules, "autoscaling_arena.adapters.mpc", fake_module)
    created["mpc_cls"] = FakeMPC
    return created


def test_build_sim_factory_mpc_builds_fresh_capacity_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    (mpc,) = _sim_autoscalers(
        _write_sim_config(
            tmp_path,
            autoscalers="""
            - name: mpc
              type: mpc
              start: {prefill: 1, decode: 1}
            """,
        )
    )
    config_before = dict(mpc.config)
    created = _install_fake_mpc_module(monkeypatch)
    capabilities = SimpleNamespace(
        decode=SimpleNamespace(max_kv_tokens=4096, max_num_seqs=64)
    )

    targets = ((1, 1), (2, 1))
    factory = match_runner._build_sim_factory(
        mpc, topology="disagg", feasible_targets=targets
    )
    first = factory(object(), capabilities)
    second = factory(object(), capabilities)

    assert isinstance(first, created["mpc_cls"]) and first is not second
    assert len(created["ais"]) == 2 and not created["roofline"]
    assert created["ais"][0].kwargs == {
        "config": EXPECTED_AIS_SPEC["config"],
        "decode_config": EXPECTED_AIS_SPEC["decode_config"],
        "max_kv_tokens": 4096,
        "max_num_seqs": 64,
    }
    assert first.kwargs["capacity_model"] is created["ais"][0]
    assert first.kwargs["feasible_targets"] == targets
    assert (
        first.kwargs["mode"] == "disagg"
        and first.kwargs["capabilities"] is capabilities
    )
    assert first.kwargs["max_replicas"] == 8 and first.kwargs["gpus_per_worker"] == 1
    assert "forward_model" not in first.kwargs
    # The resolved Match Config is reporting metadata and must stay intact.
    assert mpc.config == config_before


def test_build_sim_factory_mpc_roofline_needs_no_engine_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    (mpc,) = _sim_autoscalers(
        _write_sim_config(
            tmp_path,
            topology="agg",
            autoscalers="""
            - name: mpc-lite
              type: mpc
              start: {decode: 1}
              config: {forward_model: roofline}
            """,
        )
    )
    created = _install_fake_mpc_module(monkeypatch)
    instance = match_runner._build_sim_factory(mpc, topology="agg")(None, None)
    assert len(created["roofline"]) == 1 and not created["ais"]
    assert instance.kwargs["capacity_model"] is created["roofline"][0]
    assert instance.kwargs["mode"] == "agg"


def test_mpc_reads_the_canonical_nextn_and_an_explicit_cold_start(tmp_path: Path):
    config = yaml.safe_load(MPC_EXAMPLE.read_text())
    backend = config["backend"]
    backend["planner_config"] = str(PLANNER_CONFIG)
    backend["engines"]["common"]["extra_args"]["aic_nextn"] = 2
    backend["engines"]["common"]["runtime"]["cold_start_delay_s"] = 45
    entry = next(a for a in backend["autoscalers"] if a["type"] == "mpc")
    entry["config"]["cold_start_s"] = 45
    path = tmp_path / "match.yaml"
    path.write_text(yaml.safe_dump(config))

    mpc = next(a for a in _sim_autoscalers(path) if a.type == "mpc")
    # Both sessions predict with the speculative-decoding setting the engine runs.
    assert mpc.config["forward_model"]["config"]["nextn"] == 2
    assert mpc.config["forward_model"]["decode_config"]["nextn"] == 2
    assert mpc.config["cold_start_s"] == 45


def test_mpc_rejects_disaggregated_settings_in_aggregated_topology(tmp_path: Path):
    with pytest.raises(MatchConfigError, match="unused for agg topology"):
        load_match_config(
            _write_sim_config(
                tmp_path,
                topology="agg",
                autoscalers="""
                - name: bad
                  type: mpc
                  start: {decode: 1}
                  config: {kv_transfer_gbps: 100}
                """,
            )
        )


# --------------------------------------------------------------------------- #
# Real adapters from the example configs (need Dynamo's planner types)
# --------------------------------------------------------------------------- #


def test_mpc_example_builds_the_real_controller():
    pytest.importorskip("dynamo.planner.core.types")
    from autoscaling_arena.adapters.mpc import AISCapacityModel, MPCAutoscaler

    backend = load_match_config(MPC_EXAMPLE).backend
    entry = next(a for a in backend.autoscalers if a.type == "mpc")
    factory = match_runner._build_sim_factory(
        entry,
        topology=backend.topology,
        feasible_targets=feasible_worker_targets(backend, entry),
        short_output_itl="fail",
    )
    # Every resolved setting must be one the controller's constructor accepts.
    controller = factory(None, SimpleNamespace(decode=None))
    assert isinstance(controller, MPCAutoscaler)
    assert isinstance(controller._capacity, AISCapacityModel)
    assert controller._fleet_targets == tuple((0, n) for n in range(1, 9))
    assert controller._cold_start_s == 30


def test_cognitive_agent_example_builds_the_joint_fleet_policy(tmp_path: Path):
    pytest.importorskip("dynamo.planner.core.types")
    pytest.importorskip("httpx")
    from autoscaling_arena.adapters.jev import JevAutoscaler

    backend = load_match_config(
        ARENA_ROOT / "configs" / "match.cognitive-agent.example.yaml"
    ).backend
    entry = backend.autoscalers[0]
    # The context the Match runner supplies: no separate worker-count limit.
    context = {
        "slo": {"ttft_ms": 2000.0, "itl_ms": 50.0, "e2e_ms": None},
        "short_output_itl": "fail",
        "gpu_budget": backend.gpu_budget,
        "engines": {
            role: {"gpus_per_worker": 1, "cold_start_delay_s": 60.0}
            for role in ("prefill", "decode")
        },
    }
    factory = match_runner._build_sim_factory(
        entry,
        topology=backend.topology,
        decision_log=tmp_path / "decisions.jsonl",
        jev_context=context,
    )
    agent = factory(None, None)
    try:
        assert isinstance(agent, JevAutoscaler) and agent.consumes_replay_telemetry
        # Fleets are bounded by the pool maxima (16 prefill, 8 decode) and 32 GPUs.
        targets = agent._knowline.targets
        assert (16, 8) in targets and (16, 9) not in targets
        assert all(p + d <= backend.gpu_budget for p, d in targets)
    finally:
        asyncio.run(agent.shutdown())
