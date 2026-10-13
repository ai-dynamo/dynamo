# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cluster-shared native G2 (host) KV offload through Dynamo offline replay."""

import json

import pytest

from dynamo.mocker.config import normalize_mocker_config
from dynamo.replay import run_trace_replay

from .replay_utils import _report_summary

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.timeout(120),
]

BLOCK_SIZE = 4


def _engine_args(native_host_offload, num_gpu_blocks):
    return normalize_mocker_config(
        {
            "engine": {
                "block_size": BLOCK_SIZE,
                "num_gpu_blocks": num_gpu_blocks,
                "max_num_seqs": 1,
                "max_num_batched_tokens": 64,
                "speedup_ratio": 1000.0,
                "kv_cache_bytes_per_token": 1024,
                "native_host_offload": native_host_offload,
            }
        }
    )


def _write_trace(tmp_path, prompts):
    trace_path = tmp_path / "trace.jsonl"
    rows = [
        {
            "timestamp": 100.0 * index,
            "input_length": 10,
            "output_length": 1,
            "hash_ids": hash_ids,
        }
        for index, hash_ids in enumerate(prompts)
    ]
    trace_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return trace_path


def _host_reuse(report):
    return [
        record["first_admission_host_reused_input_tokens"]
        for record in report.per_request
    ]


def _replay(trace_path, engine_args, num_workers, router_mode):
    report = run_trace_replay(
        trace_path,
        extra_engine_args=engine_args,
        num_workers=num_workers,
        replay_mode="offline",
        router_mode=router_mode,
        trace_block_size=BLOCK_SIZE,
        capture_per_request=True,
    )
    summary = _report_summary(report)
    assert summary["completed_requests"] == summary["num_requests"]
    assert len(report.per_request) == summary["num_requests"]
    return report


def test_cluster_shared_g2_is_reachable_from_every_worker(tmp_path):
    host_offload = {
        "scope": "cluster_shared",
        "num_host_blocks": 64,
        "kv_layout_id": "test-tp1",
    }
    trace_path = _write_trace(tmp_path, [[1, 2, 3], [1, 2, 3]])

    # Round-robin forces the repeat onto the worker that never computed it.
    round_robin = _replay(trace_path, _engine_args(host_offload, 16), 2, "round_robin")
    assert _host_reuse(round_robin) == [0, 8]
