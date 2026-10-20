# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402

"""Installed native replay must execute the selected AIS speculative scheme."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

pytest.importorskip("aisimulate.sweeper")

from aisimulate.sweeper.provider import AdapterReplaySpec, RuntimeHookSpec
from aisimulate.sweeper.replay import (
    BackendDeploymentSpec,
    ReplayOutputRequirements,
    ReplaySpec,
)

from dynamo.replay import run_synthetic_trace_replay
from dynamo.replay.simulation import DynamoReplayRunnerFactory

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.integration,
    pytest.mark.gpu_0,
    pytest.mark.planner,
    pytest.mark.timeout(60),
]

_FIXTURES = Path(__file__).parent / "e2e/configs/unified_cli/fixtures"


@pytest.fixture
def draft_checkpoint(tmp_path):
    # Geometry-only test: no model weights, network access, or GPU allocation.
    config = json.loads((_FIXTURES / "tiny-model/config.json").read_text())
    config.update(
        num_hidden_layers=1,
        block_size=8,
        target_layer_ids=[0, 1, 2],
        dflash_config={"target_layer_ids": [0, 1, 2]},
        markov_rank=256,
    )
    draft = tmp_path / "draft"
    draft.mkdir()
    (draft / "config.json").write_text(json.dumps(config))
    return str(draft), config


def _selection(kind, draft_checkpoint):
    params = {
        "mtp": {"depth": 3},
        "ngram": {"num_speculative_tokens": 3},
        "eagle3": {"num_speculative_tokens": 3},
        "dflash": {"num_draft_tokens": 3},
        "dspark": {"num_draft_tokens": 3},
        "draft_model": {"num_speculative_tokens": 3},
    }[kind]
    if kind == "ngram":
        return {
            "kind": kind,
            "num_speculative_tokens": 3,
            "acceptance_rates": [1.0, 1.0, 0.4],
            "seed": 73,
        }
    selection = {
        "kind": kind,
        "params": params,
        "expected_accepted_tokens": 2.4,
        "seed": 73,
    }
    if kind != "mtp":
        path, config = draft_checkpoint
        selection.update(draft_model_path=path, draft_config=config)
    return selection


def _engine_args(backend, role, selection):
    return {
        "engine_type": backend,
        "worker_type": role,
        "block_size": 4,
        "num_gpu_blocks": 256,
        "speculation": copy.deepcopy(selection),
        "timing_model": {
            "type": "external",
            "provider": "aic",
            "config": {
                "model": str((_FIXTURES / "tiny-model").resolve()),
                "system": "h200_sxm",
                "backend": backend,
                "backend_version": "0.24.0" if backend == "vllm" else "0.5.14",
                "worker_type": role,
                "tp": 1,
                "estimation_mode": "op_level",
            },
        },
    }


def _spec(backend, topology, selection, trace):
    if topology == "agg":
        engines = {
            "agg_engine_args": _engine_args(backend, "aggregated", selection),
            "num_workers": 1,
        }
    else:
        engines = {
            "prefill_engine_args": _engine_args(backend, "prefill", selection),
            "decode_engine_args": _engine_args(backend, "decode", selection),
            "num_prefill_workers": 1,
            "num_decode_workers": 1,
        }
    return ReplaySpec(
        backend_deployment=BackendDeploymentSpec(
            deployment_mode=topology,
            backend=backend,
            backend_version="0.24.0" if backend == "vllm" else "0.5.14",
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
                        config={
                            "router_mode": "kv_router"
                            if backend == "sglang"
                            else "round_robin",
                            "router_config": {},
                        },
                    ),
                )
            )
        },
    )


@pytest.mark.parametrize("topology", ["agg", "disagg"])
@pytest.mark.parametrize(
    "kind,backend",
    [
        ("mtp", "vllm"),
        ("mtp", "sglang"),
        ("eagle3", "vllm"),
        ("eagle3", "sglang"),
        ("ngram", "vllm"),
        ("dflash", "vllm"),
        ("dspark", "sglang"),
        ("draft_model", "vllm"),
    ],
)
def test_agentic_replay_preserves_scheme_and_accepted_progress(
    tmp_path, draft_checkpoint, kind, backend, topology
):
    rows = [
        json.loads(line)
        for line in (_FIXTURES / "traces/agentic-mooncake.jsonl")
        .read_text()
        .splitlines()
    ]
    for row in rows[1:]:
        row["output_length"] = 32
    trace = tmp_path / "agentic.jsonl"
    trace.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    selection = _selection(kind, draft_checkpoint)
    spec = _spec(backend, topology, selection, trace)
    runner = DynamoReplayRunnerFactory().create(0)
    role = "aggregated" if topology == "agg" else "decode"
    native = runner._engine_args(spec.backend_deployment, role)["engine"]
    assert native["timing_model"]["config"]["speculation"]["kind"] == kind
    assert native["aic_nextn"] == 3
    assert native["aic_mtp_seed"] == 73
    assert list(
        map(float, native["aic_nextn_accept_rates"].split(","))
    ) == pytest.approx([1.0, 1.0, 0.4])
    output = ReplayOutputRequirements(capture_per_request=True, include_raw_report=True)
    try:
        report = runner.run(spec, output_requirements=output)
        assert report.metrics["completed_requests"] == 3
        assert report.metrics["total_output_tokens"] == 96
        assert report.metrics["completed_trajectories"] == 1
        assert report.metrics["incomplete_trajectories"] == 0
        assert report.metadata["agentic_qualification"] == "functional_only"
        records = report.metadata["native_report"]["per_request"]
        assert len(records) == 3
        assert all(record["output_length"] == 32 for record in records)
        if kind in {"mtp", "eagle3"}:
            selection["expected_accepted_tokens"] = 0.0
            no_acceptance = runner.run(_spec(backend, topology, selection, trace))
            assert no_acceptance.metrics["total_output_tokens"] == 96
            assert report.metrics["duration_ms"] < no_acceptance.metrics["duration_ms"]
    finally:
        runner.close()


@pytest.mark.parametrize(
    "invalid", ["capacity", "inferred", "rates", "depth", "tree", "conflict"]
)
def test_native_boundary_rejects_invalid_speculative_controls(
    draft_checkpoint, invalid
):
    selection = _selection("eagle3", draft_checkpoint)
    spec = _spec("vllm", "agg", selection, "unused.jsonl")
    runner = DynamoReplayRunnerFactory().create(0)
    native = runner._engine_args(spec.backend_deployment, "aggregated")
    engine = native["engine"]
    cost = engine["timing_model"]["config"]["speculation"]
    if invalid in {"capacity", "inferred"}:
        if invalid == "capacity":
            engine.pop("num_gpu_blocks")
            native["num_gpu_blocks_is_explicit"] = False
        else:
            native["num_gpu_blocks_is_explicit"] = False
        message = "explicitly configured positive num_gpu_blocks"
    elif invalid == "rates":
        engine.pop("aic_nextn_accept_rates")
        message = "explicit aic_nextn_accept_rates"
    elif invalid == "tree":
        cost["params"] = {"tree_shape": [1, 2]}
        message = "chain"
    elif invalid == "depth":
        cost["params"] = {"num_speculative_tokens": 6}
        message = "1.*5"
    else:
        engine["aic_nextn"] = 2
        engine["aic_nextn_accept_rates"] = "1,0.5"
        message = "canonical speculative depth conflicts"
    with pytest.raises(ValueError, match=message):
        run_synthetic_trace_replay(
            8, 8, 1, extra_engine_args=native, replay_concurrency=1
        )
