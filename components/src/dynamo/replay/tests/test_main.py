# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Optional-dependency preflight must run before replay CLI imports.
# ruff: noqa: E402

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip(
    "aisimulate.replay",
    reason="AI Simulate is an optional Dynamo simulation dependency",
)

from aisimulate import aic
from aisimulate.replay import cli as engine_replay_cli
from aisimulate.replay.config import parse_base_replay_config
from aisimulate.runner import EngineReplayRunnerFactory
from aisimulate.sweeper.replay import canonical_json
from dynamo.replay import ReplayReport
from dynamo.replay import main as replay_main
from dynamo.replay import reporting as replay_reporting

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.planner,
]


def _stub_cli_dependencies(monkeypatch, seen: dict) -> None:
    monkeypatch.setattr(replay_main, "_load_engine_args", lambda value: value)
    monkeypatch.setattr(
        replay_main,
        "_load_router_config",
        lambda config, policy: (config, policy),
    )
    monkeypatch.setattr(replay_main, "_load_aic_perf_config", lambda args: None)
    monkeypatch.setattr(
        replay_main,
        "write_report_json",
        lambda payload, path: seen.setdefault("report_payload", payload) or path,
    )
    monkeypatch.setattr(replay_main, "format_report_table", lambda summary: "table")


def test_offline_cli_serializes_report_and_per_request_jsonl(
    monkeypatch, tmp_path
) -> None:
    seen: dict = {}
    _stub_cli_dependencies(monkeypatch, seen)

    def run_synthetic(*args, **kwargs):
        seen["native_args"] = args
        seen["native_kwargs"] = kwargs
        return ReplayReport(
            summary={"completed_requests": 1},
            per_request=[{"request_id": "request-1"}],
            coverage={"captured_request_count": 1},
            planner=None,
        )

    monkeypatch.setattr(replay_main, "run_synthetic_trace_replay", run_synthetic)
    per_request_path = tmp_path / "requests.jsonl"

    assert (
        replay_main.main(
            [
                "--input-tokens",
                "8",
                "--output-tokens",
                "4",
                "--request-count",
                "1",
                "--replay-concurrency",
                "1",
                "--per-request-jsonl",
                str(per_request_path),
            ]
        )
        == 0
    )

    assert seen["native_kwargs"]["capture_per_request"] is True
    assert "agentic_lanes" not in seen["native_kwargs"]
    assert seen["report_payload"] == {
        "summary": {"completed_requests": 1},
        "per_request": [{"request_id": "request-1"}],
        "coverage": {"captured_request_count": 1},
        "planner": None,
    }
    assert json.loads(per_request_path.read_text()) == {"request_id": "request-1"}


def test_agentic_lanes_reject_synthetic_replay() -> None:
    args = replay_main.build_parser().parse_args(
        [
            "--input-tokens",
            "8",
            "--output-tokens",
            "4",
            "--request-count",
            "1",
            "--replay-concurrency",
            "1",
            "--agentic-lanes",
            "1",
        ]
    )

    with pytest.raises(ValueError, match="requires trace-file replay"):
        parse_base_replay_config(args)


def test_online_cli_keeps_flat_report(monkeypatch) -> None:
    seen: dict = {}
    _stub_cli_dependencies(monkeypatch, seen)

    def run_synthetic(*args, **kwargs):
        seen["native_kwargs"] = kwargs
        return {"completed_requests": 1}

    monkeypatch.setattr(replay_main, "run_synthetic_trace_replay", run_synthetic)

    assert (
        replay_main.main(
            [
                "--input-tokens",
                "8",
                "--output-tokens",
                "4",
                "--request-count",
                "1",
                "--replay-concurrency",
                "1",
                "--replay-mode",
                "online",
            ]
        )
        == 0
    )

    assert seen["native_kwargs"]["capture_per_request"] is False
    assert seen["report_payload"] == {"completed_requests": 1}


def test_reporting_is_self_contained_and_preserves_default_name(
    monkeypatch, tmp_path
) -> None:
    import_script = """
import builtins
import importlib.util
import sys

original_import = builtins.__import__

def reject_aisimulate(name, *args, **kwargs):
    if name == "aisimulate" or name.startswith("aisimulate."):
        raise ModuleNotFoundError(name)
    return original_import(name, *args, **kwargs)

builtins.__import__ = reject_aisimulate
spec = importlib.util.spec_from_file_location("dynamo_replay_reporting", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert module.TITLE == "NVIDIA AIPerf | LLM Metrics"
"""

    subprocess.run(
        [sys.executable, "-I", "-c", import_script, replay_reporting.__file__],
        check=True,
        capture_output=True,
        text=True,
    )

    monkeypatch.chdir(tmp_path)
    report_path = replay_reporting.write_report_json({"completed_requests": 1}, None)

    assert "Request Count (requests)" in replay_reporting.format_report_table(
        {"completed_requests": 1}
    )
    assert report_path.parent == tmp_path
    assert report_path.name.startswith("dynamo_replay_report_")
    assert json.loads(report_path.read_text()) == {"completed_requests": 1}


def test_cli_reports_missing_aisimulate_clearly() -> None:
    entrypoint = Path(__file__).resolve().parents[1] / "__main__.py"
    import_script = r"""
import builtins
import runpy
import sys

original_import = builtins.__import__

def reject_aisimulate(name, *args, **kwargs):
    if name == "aisimulate" or name.startswith("aisimulate."):
        raise ModuleNotFoundError(
            f"No module named {name!r}",
            name=name,
        )
    return original_import(name, *args, **kwargs)

builtins.__import__ = reject_aisimulate
try:
    runpy.run_path(sys.argv[1], run_name="__main__")
except SystemExit as exc:
    assert str(exc) == (
        "Dynamo replay requires the 'aisimulate' package and currently "
        "supports Python 3.10-3.12."
    )
else:
    raise AssertionError("Dynamo replay unexpectedly loaded without AISimulate")
"""

    subprocess.run(
        [sys.executable, "-I", "-c", import_script, str(entrypoint)],
        check=True,
        capture_output=True,
        text=True,
    )


def test_dynamo_cli_reuses_exact_aisimulate_base_schema() -> None:
    common_argv = [
        "--extra-engine-args",
        json.dumps(
            {
                "engine_type": "vllm",
                "num_gpu_blocks": 16,
                "block_size": 4,
            }
        ),
        "--input-tokens",
        "8",
        "--output-tokens",
        "4",
        "--request-count",
        "2",
        "--arrival-interval-ms",
        "3.5",
        "--num-workers",
        "2",
        "--sla-ttft-ms",
        "20",
        "--report-json",
        "report.json",
        "--per-request-jsonl",
        "requests.jsonl",
    ]

    engine_config = parse_base_replay_config(
        engine_replay_cli.build_parser().parse_args(common_argv)
    )
    dynamo_config = parse_base_replay_config(
        replay_main.build_parser().parse_args(common_argv)
    )

    assert dynamo_config == engine_config
    assert canonical_json(dynamo_config.to_replay_spec()) == canonical_json(
        engine_config.to_replay_spec()
    )


def test_dynamo_cli_contains_every_aisimulate_base_option() -> None:
    def option_strings(parser):
        return {
            option
            for action in parser._actions
            for option in action.option_strings
            if option != "--help"
        }

    assert option_strings(engine_replay_cli.build_parser()) <= option_strings(
        replay_main.build_parser()
    )


def test_dynamo_cli_adds_router_planner_and_online_extensions() -> None:
    args = replay_main.build_parser().parse_args(
        [
            "--router-mode",
            "kv_router",
            "--planner-config",
            "planner.yaml",
            "--replay-mode",
            "online",
        ]
    )

    assert args.router_mode == "kv_router"
    assert args.planner_config == "planner.yaml"
    assert args.replay_mode == "online"


def test_engine_and_dynamo_mains_apply_identical_aic_capacity_lowering(
    monkeypatch, tmp_path
) -> None:
    estimator_calls = []

    def estimate(**kwargs):
        estimator_calls.append(kwargs)
        return 444

    monkeypatch.setattr(aic, "estimate_num_gpu_blocks", estimate)

    class RecordingRuntime:
        execution_spec = None

        def run_replay_json(self, execution_spec_json):
            self.execution_spec = json.loads(execution_spec_json)
            return json.dumps(
                {
                    "completed_requests": 1,
                    "output_throughput_tok_s": 1.0,
                }
            )

    runtime = RecordingRuntime()
    monkeypatch.setattr(
        engine_replay_cli,
        "EngineReplayRunnerFactory",
        lambda: EngineReplayRunnerFactory(runtime=runtime),
    )

    dynamo_seen = {}

    def run_synthetic(*args, **kwargs):
        dynamo_seen["kwargs"] = kwargs
        return ReplayReport(
            summary={"completed_requests": 1},
            per_request=[],
            coverage={},
            planner=None,
        )

    monkeypatch.setattr(replay_main, "_load_aic_perf_config", lambda args: None)
    monkeypatch.setattr(
        replay_main,
        "run_synthetic_trace_replay",
        run_synthetic,
    )
    monkeypatch.setattr(
        replay_main,
        "write_report_json",
        lambda payload, path: tmp_path / "dynamo-report.json",
    )
    monkeypatch.setattr(replay_main, "format_report_table", lambda summary: "table")

    engine_args = json.dumps(
        {
            "engine_type": "vllm",
            "aic_backend": "vllm",
            "aic_backend_version": "0.19.0",
            "aic_system": "h200_sxm",
            "aic_model_path": "test-model",
            "aic_tp_size": 1,
            "aic_nextn": 3,
            "block_size": 64,
            "max_num_batched_tokens": 4096,
        }
    )
    common = [
        "--extra-engine-args",
        engine_args,
        "--input-tokens",
        "8",
        "--output-tokens",
        "1",
        "--request-count",
        "1",
        "--replay-concurrency",
        "1",
    ]

    assert (
        engine_replay_cli.main(
            [*common, "--report-json", str(tmp_path / "engine-report.json")]
        )
        == 0
    )
    assert replay_main.main(common) == 0

    engine_blocks = runtime.execution_spec["engine"]["rank"]["num_gpu_blocks"]
    dynamo_blocks = dynamo_seen["kwargs"]["extra_engine_args"].num_gpu_blocks
    assert engine_blocks == dynamo_blocks == 444
    assert len(estimator_calls) == 2
    assert all("nextn" not in call for call in estimator_calls)
