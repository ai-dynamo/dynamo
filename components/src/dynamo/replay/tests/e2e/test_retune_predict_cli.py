# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402

"""Planner retune through ``aisimulate predict --stack dynamo``.

The workload is the one in ``test_retune_replay.py``: a dense 20 s head and a
sparse tail on 4 workers, where greedy routing (``router_temperature: 0``)
beats near-random routing (``100``) under a tight TTFT SLA. The CLI runs the
replay in AISimulate's supervised execution process, so this also covers the
spawned sweep workers under that supervisor. Traffic that retune cannot
replay is rejected by the replay, in AISimulate's terms.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytest.importorskip(
    "aisimulate.config.cli",
    reason="AISimulate is an optional Dynamo simulation dependency",
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.pre_merge,
    pytest.mark.planner,
    pytest.mark.gpu_0,
]

_TINY_MODEL = (
    Path(__file__).resolve().parent / "configs/unified_cli/fixtures/tiny-model"
)
_BASE_MS = 1_700_000_000_000
_HEAD_REQUESTS = 500
_TAIL_REQUESTS = 125


def _write_trace(
    path: Path, *, count: int, gap_ms: int, start_ms: int, prefix: str
) -> None:
    with path.open("w", encoding="utf-8") as sink:
        for index in range(count):
            received_ms = start_ms + index * gap_ms
            row = {
                "schema": "dynamo.request.trace.v1",
                "event_type": "request_end",
                "event_time_unix_ms": received_ms + 10,
                "request": {
                    "request_id": f"{prefix}-{index}",
                    "request_received_ms": received_ms,
                    "output_tokens": 128,
                    "replay": {
                        "trace_block_size": 64,
                        "input_length": 64 * 16,
                        "input_sequence_hashes": [
                            index * 1_000 + block for block in range(16)
                        ],
                    },
                },
            }
            sink.write(json.dumps(row) + "\n")


def _config() -> dict:
    return {
        "traffic": {
            # Relative paths resolve against the working directory.
            "source": {
                "type": "trace",
                "format": "dynamo",
                "paths": ["head.jsonl", "tail.jsonl"],
            },
            "load": {"type": "trace_timestamps"},
        },
        "engine": {
            "mode": "aggregated",
            "model": str(_TINY_MODEL),
            "hardware": "h200_sxm",
            "backend": "vllm",
            "context_length": 4096,
            "workers": {
                "aggregated": {
                    "parallelism": {
                        "replicas": 4,
                        "tensor": 1,
                        "pipeline": 1,
                        "attention_data": 1,
                        "moe_tensor": 1,
                        "moe_expert": 1,
                    },
                    "scheduler": {"max_batched_tokens": 8192, "max_sequences": 8},
                    "kv_cache": {
                        "block_size": 64,
                        "prefix_caching": True,
                        "capacity": {"type": "fixed", "blocks": 4096},
                    },
                    "timing": {"type": "polynomial"},
                }
            },
        },
        "router": {"policy": "kv_router", "temperature": 100.0},
        "planner": {
            "policy": "enabled",
            "enable_load_scaling": False,
            "enable_throughput_scaling": False,
            "retune": {
                # Window 0 = [0, 120) s launches at the 125 s tick and is
                # decided at 130 s. The interval also caps the sweep's
                # wall-clock deadline, which covers worker start-up.
                "interval_seconds": 120,
                "grace_seconds": 1,
                "max_parallel": 2,
                # AISimulate recommend syntax: credit and scale pinned at
                # the router's values, temperature searched.
                "router": {
                    "overlap_score_credit": 1.0,
                    "prefill_load_scale": 1.0,
                    "temperature": {"choices": [0.0, 100.0]},
                },
            },
        },
        # Also the retune scoring SLA (planner.retune sets none).
        "evaluation": {"sla": {"ttft_ms": 60.0, "itl_ms": 100.0}},
    }


def _predict(tmp_path: Path, config: dict) -> subprocess.CompletedProcess[str]:
    (tmp_path / "retune.yaml").write_text(yaml.safe_dump(config))
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "aisimulate",
            "predict",
            "--stack",
            "dynamo",
            "--config",
            "retune.yaml",
            "--output-dir",
            "out",
            "--format",
            "json",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=170,
        check=False,
    )


@pytest.mark.timeout(180)  # ~20 s measured; spawned sweep workers vary with load
def test_predict_cli_reports_retune_decisions(tmp_path: Path) -> None:
    _write_trace(
        tmp_path / "head.jsonl",
        count=_HEAD_REQUESTS,
        gap_ms=40,
        start_ms=_BASE_MS,
        prefix="request",
    )
    _write_trace(
        tmp_path / "tail.jsonl",
        count=_TAIL_REQUESTS,
        gap_ms=1_000,
        start_ms=_BASE_MS + _HEAD_REQUESTS * 40,
        prefix="tail",
    )

    result = _predict(tmp_path, _config())
    assert result.returncode == 0, result.stderr[-4000:]

    runtime = json.loads((tmp_path / "out/resource-runtime.json").read_text())
    assert runtime["status"] == "completed", runtime
    report = json.loads((tmp_path / "out/prediction.json").read_text())
    assert report["summary"]["completed_requests"] == _HEAD_REQUESTS + _TAIL_REQUESTS
    planner = report["planner"]
    assert planner["scaling_events"] == []

    first = planner["retune_events"][0]
    assert first["window_index"] == 0
    assert (first["launched_at_s"], first["decided_at_s"]) == (125.0, 130.0)
    assert first["outcome"] == "applied", first
    pinned = {"overlap_score_credit": 1.0, "prefill_load_scale": 1.0}
    assert first["previous"]["router"] == {**pinned, "router_temperature": 100.0}
    assert first["chosen"]["router"] == {**pinned, "router_temperature": 0.0}
    assert len(first["top"]) == 2
    assert first["best_score"] > first["baseline_score"]

    # The replay router takes the new temperature from the decision tick on.
    for tick in planner["ticks"]:
        desired = tick["runtime_decision"]["router_parameters"]["aggregated"]
        expected = 0.0 if tick["at_ms"] >= 130_000.0 else 100.0
        assert desired["router_temperature"] == expected


@pytest.mark.timeout(180)  # ~4 s measured: the replay rejects at startup
def test_predict_cli_rejects_traffic_retune_cannot_replay(tmp_path: Path) -> None:
    # The other rejected inputs are unit-tested on _retune_inputs in
    # test_retune_replay.py; this proves such an error reaches the CLI.
    _write_trace(
        tmp_path / "head.jsonl", count=20, gap_ms=40, start_ms=_BASE_MS, prefix="r"
    )
    config = _config()
    config["traffic"] = {
        **config["traffic"],
        "source": {**config["traffic"]["source"], "paths": ["head.jsonl"]},
        "load": {"type": "concurrency", "concurrency": 4},
    }

    result = _predict(tmp_path, config)

    assert result.returncode != 0
    assert (
        "planner.retune" in result.stderr
        and "traffic.load.type: concurrency" in result.stderr
    ), result.stderr[-4000:]
