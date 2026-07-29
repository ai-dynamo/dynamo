# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused ownership tests for Replay planner construction and teardown."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import ANY

import pytest

import dynamo.replay.main as replay_main

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def test_report_namespace_component_is_stable_and_collision_resistant() -> None:
    slash = replay_main._report_namespace_component("model/a")
    question = replay_main._report_namespace_component("model?a")
    upper = replay_main._report_namespace_component("Deployment")
    lower = replay_main._report_namespace_component("deployment")
    unicode_only = replay_main._report_namespace_component("模型")

    assert replay_main._report_namespace_component(None) is None
    assert slash == replay_main._report_namespace_component("model/a")
    assert slash is not None and slash.startswith("model_a-")
    assert question is not None and question.startswith("model_a-")
    assert slash != question
    assert upper is not None and lower is not None
    assert upper.casefold() != lower.casefold()
    assert unicode_only is not None
    assert unicode_only.startswith("deployment-")


class _FakePlannerConfig:
    mode = "agg"
    optimization_target = "sla"

    def __init__(self, report_output_dir: Path) -> None:
        self.advisory = False
        self.report_output_dir = str(report_output_dir)


class _FakePlannerReplayBridge:
    @classmethod
    def from_synthetic(cls, **_kwargs: Any) -> _FakePlannerReplayBridge:
        return cls()


class _FakeWorkerCapabilities:
    def __init__(self, **_kwargs: Any) -> None:
        pass


def _patch_builder_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    adapter_type: type,
) -> None:
    import dynamo._internal.aic as aic_module
    import dynamo.mocker as mocker_module
    import dynamo.planner.config.planner_config as config_module
    import dynamo.planner.core.types as types_module
    import dynamo.planner.offline.replay_adapter as adapter_module

    config = _FakePlannerConfig(tmp_path)

    class _ConfigFactory:
        @staticmethod
        def from_config_arg(_config_arg: str) -> _FakePlannerConfig:
            return config

    monkeypatch.setattr(config_module, "PlannerConfig", _ConfigFactory)
    monkeypatch.setattr(
        mocker_module,
        "PlannerReplayBridge",
        _FakePlannerReplayBridge,
        raising=False,
    )
    monkeypatch.setattr(types_module, "WorkerCapabilities", _FakeWorkerCapabilities)
    monkeypatch.setattr(adapter_module, "ReplayPlannerAdapter", adapter_type)
    monkeypatch.setattr(replay_main, "_engine_caps", lambda _args: object())
    monkeypatch.setattr(aic_module, "create_session", lambda **_kwargs: object())
    monkeypatch.setattr(
        replay_main,
        "_generate_aic_prefill_fpms",
        lambda *_args, **_kwargs: [object()],
    )
    monkeypatch.setattr(
        replay_main,
        "_generate_aic_decode_fpms",
        lambda *_args, **_kwargs: [object()],
    )


def _aic_engine_args() -> SimpleNamespace:
    return SimpleNamespace(
        aic_backend="test",
        aic_system="test-system",
        aic_model_path="test-model",
        aic_tp_size=1,
        aic_backend_version=None,
        aic_moe_tp_size=None,
        aic_moe_ep_size=None,
        aic_attention_dp_size=None,
        aic_gemm_dtype=None,
        aic_moe_dtype=None,
        aic_fmha_dtype=None,
        aic_kv_cache_dtype=None,
        aic_comm_dtype=None,
        aic_nextn=None,
    )


def test_builder_closes_adapter_after_regression_install_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class RegressionInstallError(RuntimeError):
        pass

    class _FailingAdapter:
        instances: list[_FailingAdapter] = []

        def __init__(self, **kwargs: Any) -> None:
            self.event_loop = kwargs["event_loop"]
            self.close_calls = 0
            self.installed: dict[str, Any] | None = None
            self.instances.append(self)

        def install_benchmark_fpms(self, **kwargs: Any) -> None:
            self.installed = kwargs
            raise RegressionInstallError("regression install failed")

        def close(self) -> None:
            self.close_calls += 1

    _patch_builder_dependencies(monkeypatch, tmp_path, _FailingAdapter)
    shared_loop = asyncio.new_event_loop()
    try:
        with pytest.raises(RegressionInstallError, match="regression install failed"):
            replay_main._build_planner_replay(
                trace_file=None,
                extra_engine_args=_aic_engine_args(),
                prefill_engine_args=None,
                decode_engine_args=None,
                router_config=None,
                num_workers=1,
                num_prefill_workers=0,
                num_decode_workers=0,
                router_mode="round_robin",
                arrival_speedup_ratio=1.0,
                trace_block_size=512,
                planner_config_arg="ignored",
                synthetic=replay_main.SyntheticWorkload(
                    input_tokens=1,
                    output_tokens=1,
                    request_count=1,
                ),
                event_loop=shared_loop,
            )

        adapter = _FailingAdapter.instances[-1]
        assert adapter.event_loop is shared_loop
        assert adapter.close_calls == 1
        assert adapter.installed == {
            "agg_fpms": [ANY, ANY],
        }
        assert not shared_loop.is_closed()
    finally:
        shared_loop.close()


def test_legacy_run_preserves_run_error_when_cleanup_also_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RunError(RuntimeError):
        pass

    class _Bridge:
        def run(self, _adapter: object) -> None:
            raise RunError("planner callback failed")

    class _Adapter:
        def close(self) -> None:
            raise RuntimeError("transport shutdown failed")

    monkeypatch.setattr(
        replay_main,
        "_build_planner_replay",
        lambda *_args, **_kwargs: (_Bridge(), _Adapter()),
    )

    with pytest.raises(RunError, match="planner callback failed") as exc_info:
        replay_main._run_planner_replay()

    assert exc_info.value.__notes__ == [
        "Replay planner cleanup also failed: transport shutdown failed"
    ]
