# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Match Config and factory wiring of the CloudAI MPC adapter, plus the lazy exports.

Pure Python: no Dynamo, numpy or torch is needed. Adapter behaviour is covered
by ``test_cloudai_mpc_v3.py`` and ``test_cloudai_rl_lstm_disagg.py``.
"""

from __future__ import annotations

import importlib
import sys
import textwrap
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from autoscaling_arena import match_runner
from autoscaling_arena.match_config import (
    MatchConfigError,
    SimAutoscalerConfig,
    SimBackendConfig,
    load_match_config,
)

ARENA_ROOT = Path(__file__).resolve().parents[1]
PLANNER_CONFIG = ARENA_ROOT / "configs" / "planner.sim.example.yaml"
CLOUDAI_EXAMPLE = ARENA_ROOT / "configs" / "match.sim.cloudai.example.yaml"

# ``substrate: gpt_oss`` pins the canonical AIS identity of the prefill engine.
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


def _write_sim_config(
    tmp_path: Path, *, autoscalers: str, topology: str = "disagg", gpu_budget: int = 32
) -> Path:
    entries = textwrap.indent(textwrap.dedent(autoscalers).strip(), "    ")
    body = textwrap.dedent(
        f"""
        schema_version: 1
        name: cloudai-contract
        backend:
          type: sim
          substrate: gpt_oss
          topology: {topology}
          gpu_budget: {gpu_budget}
          planner_config: {PLANNER_CONFIG}
          autoscalers:
        @@AUTOSCALERS@@
        evaluations:
          workloads: [flat]
          defaults:
            seed: 0
            max_requests: 50
            arrival_speedup: 1.0
        slo_profiles:
          - name: relaxed
            ttft_ms: 2000
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
    path.write_text(body.replace("@@AUTOSCALERS@@", entries))
    return path


def _sim_autoscalers(path: Path) -> tuple[SimAutoscalerConfig, ...]:
    backend = load_match_config(path).backend
    assert isinstance(backend, SimBackendConfig)
    return backend.autoscalers


# --------------------------------------------------------------------------- #
# Match Config
# --------------------------------------------------------------------------- #


def test_cloudai_mpc_defaults_resolve_forward_model_from_backend(tmp_path: Path):
    (mpc,) = _sim_autoscalers(
        _write_sim_config(
            tmp_path,
            autoscalers="""
            - name: mpc
              type: cloudai_mpc_v3
              start: {prefill: 1, decode: 1}
            """,
        )
    )
    assert mpc.type == "cloudai_mpc_v3"
    assert mpc.config["forward_model"] == EXPECTED_AIS_SPEC
    assert mpc.config["gpus_per_worker"] == 1
    assert mpc.config["prefill_gpus_per_worker"] == 1  # disagg: prefill pool cost
    assert mpc.config["max_replicas"] == 8 and mpc.config["poll_interval_s"] == 5.0
    assert mpc.start.prefill == 1 and mpc.start.decode == 1


def test_cloudai_mpc_roofline_in_agg_topology_resolves_without_engine_spec(
    tmp_path: Path,
):
    (mpc,) = _sim_autoscalers(
        _write_sim_config(
            tmp_path,
            topology="agg",
            autoscalers="""
            - name: mpc-lite
              type: cloudai_mpc_v3
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
              type: cloudai_mpc_v3
              start: {prefill: 1, decode: 1}
              config: {min_prefill: 1}
            """,
            ("unknown fields", "min_prefill"),
        ),
        (
            """
            - name: bad
              type: cloudai_mpc_v3
              start: {prefill: 1, decode: 1}
              config: {smoothing_alpha: 1.5}
            """,
            ("smoothing_alpha", "<= 1"),
        ),
        (
            """
            - name: bad
              type: cloudai_mpc_v3
              start: {prefill: 1, decode: 1}
              config: {min_replicas: 5, max_replicas: 2}
            """,
            ("min_replicas must be <= max_replicas",),
        ),
        (
            """
            - name: bad
              type: cloudai_mpc_v3
              start: {prefill: 1, decode: 9}
            """,
            ("start.decode", "[1, 8]"),
        ),
        (
            """
            - name: bad
              type: cloudai_mpc_v3
              start: {prefill: 1, decode: 1}
              config: {max_replicas: 40}
            """,
            ("maximum fleet requires 80 GPUs", "gpu_budget=32", "max_replicas"),
        ),
    ],
)
def test_cloudai_config_errors(tmp_path: Path, autoscaler: str, message_parts):
    with pytest.raises(MatchConfigError) as excinfo:
        load_match_config(_write_sim_config(tmp_path, autoscalers=autoscaler))
    for part in message_parts:
        assert part in str(excinfo.value)


def test_unknown_autoscaler_type_lists_cloudai_choices(tmp_path: Path):
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
    assert (
        "cloudai_mpc_v3" in message
        and "cloudai_rl_lstm" in message
        and "cloudai_rl_disagg" in message
    )
    assert "cloudai_mpc_v2" not in message and "cloudai_rl_v3" not in message


def test_committed_cloudai_example_expands_the_roster():
    config = load_match_config(CLOUDAI_EXAMPLE)
    backend = config.backend
    assert isinstance(backend, SimBackendConfig)
    types_by_name = {entry.name: entry.type for entry in backend.autoscalers}
    assert types_by_name["cloudai-mpc"] == "cloudai_mpc_v3"
    assert types_by_name["cloudai-rl"] == "cloudai_rl_lstm"
    assert set(types_by_name.values()) == {
        "planner",
        "keda",
        "reactive",
        "static",
        "cloudai_mpc_v3",
        "cloudai_rl_lstm",
    }
    assert backend.topology == "agg" and backend.replay.telemetry_sample_interval_s == 5
    assert config.expected_runs == len(list(config.iter_runs())) == 48


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

    fake_module = types.ModuleType("autoscaling_arena.adapters.cloudai_mpc_v3")
    fake_module.AISCapacityModel = FakeAISCapacityModel  # type: ignore[attr-defined]
    fake_module.RooflineCapacityModel = FakeRooflineCapacityModel  # type: ignore[attr-defined]
    fake_module.CloudAIMPCV3Autoscaler = FakeMPC  # type: ignore[attr-defined]
    fake_adapters = types.ModuleType("autoscaling_arena.adapters")
    fake_adapters.cloudai_mpc_v3 = fake_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "autoscaling_arena.adapters", fake_adapters)
    monkeypatch.setitem(
        sys.modules, "autoscaling_arena.adapters.cloudai_mpc_v3", fake_module
    )
    created["mpc_cls"] = FakeMPC
    return created


def test_build_sim_factory_cloudai_mpc_builds_fresh_capacity_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    (mpc,) = _sim_autoscalers(
        _write_sim_config(
            tmp_path,
            autoscalers="""
            - name: mpc
              type: cloudai_mpc_v3
              start: {prefill: 1, decode: 1}
            """,
        )
    )
    config_before = dict(mpc.config)
    created = _install_fake_mpc_module(monkeypatch)
    capabilities = SimpleNamespace(
        decode=SimpleNamespace(max_kv_tokens=4096, max_num_seqs=64)
    )

    factory = match_runner._build_sim_factory(mpc, topology="disagg")
    first = factory(object(), capabilities)
    second = factory(object(), capabilities)

    assert isinstance(first, created["mpc_cls"]) and first is not second
    assert len(created["ais"]) == 2 and not created["roofline"]
    assert created["ais"][0].kwargs == {
        "config": EXPECTED_AIS_SPEC["config"],
        "max_kv_tokens": 4096,
        "max_num_seqs": 64,
    }
    assert first.kwargs["capacity_model"] is created["ais"][0]
    assert (
        first.kwargs["mode"] == "disagg"
        and first.kwargs["capabilities"] is capabilities
    )
    assert first.kwargs["max_replicas"] == 8 and first.kwargs["gpus_per_worker"] == 1
    assert "forward_model" not in first.kwargs
    # The resolved Match Config is reporting metadata and must stay intact.
    assert mpc.config == config_before


def test_build_sim_factory_cloudai_mpc_roofline_needs_no_engine_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    (mpc,) = _sim_autoscalers(
        _write_sim_config(
            tmp_path,
            topology="agg",
            autoscalers="""
            - name: mpc-lite
              type: cloudai_mpc_v3
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


# --------------------------------------------------------------------------- #
# Lazy exports
# --------------------------------------------------------------------------- #


def test_lazy_exports_do_not_import_torch_until_used(monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("dynamo.planner.core.types")
    for name in [m for m in sys.modules if m.startswith("autoscaling_arena.adapters")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "torch", None)  # make `import torch` fail

    adapters = importlib.import_module("autoscaling_arena.adapters")

    for name in (
        "CloudAIMPCV3Autoscaler",
        "CloudAIRLAutoscaleLSTM",
        "CloudAIRLDisaggAutoscaler",
    ):
        assert name in adapters.__all__ and name in dir(adapters)
    assert "autoscaling_arena.adapters.cloudai_rl_lstm" not in sys.modules
    with pytest.raises(ImportError):
        adapters.CloudAIRLAutoscaleLSTM  # noqa: B018 — the access is the test
    with pytest.raises(AttributeError):
        adapters.NotAnAdapter  # noqa: B018
