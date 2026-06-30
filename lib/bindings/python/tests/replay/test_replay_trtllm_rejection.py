# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end replay regression for TRT-LLM terminal-rejection propagation.

A no-evict request whose to-completion footprint exceeds the entire KV pool can
never be admitted, so the scheduler terminally rejects it. The rejection must
propagate a terminal signal through the replay driver, otherwise the oversized
request stalls the FIFO head and the run hangs on a never-freed `max_in_flight`
slot (the failure mode that kept this path disabled in PR #10193). The rejected
request is excluded from the report; the valid follower behind it completes.

The live (online) path is covered deterministically by the Rust unit test
`scheduler ... trtllm_oversized_request_rejected_unblocks_follower_live`; online
replay is intentionally not exercised here because of the pre-existing
intermittent online-mode hang tracked in #9548.
"""

import json

import pytest

from dynamo.mocker import MockEngineArgs
from dynamo.replay import run_trace_replay

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.timeout(60),
]


def _trtllm_reject_args():
    # 8 GPU blocks * block_size 64 = 512-token to-completion budget per request.
    return MockEngineArgs.from_json(
        json.dumps(
            {
                "engine_type": "trtllm",
                "block_size": 64,
                "num_gpu_blocks": 8,
                "max_num_seqs": 4,
                "max_num_batched_tokens": 4096,
                "enable_prefix_caching": False,
                "enable_chunked_prefill": True,
                "speedup_ratio": 1000.0,
            }
        )
    )


def _write_oversized_then_valid_trace(tmp_path):
    trace_path = tmp_path / "oversized_then_valid.jsonl"
    records = [
        # Oversized: 600-token prompt -> ceil((600 + 10) / 64) = 10 blocks > the
        # 8-block pool, so it can never be admitted and must be terminally rejected.
        {
            "timestamp": 0.0,
            "input_length": 600,
            "output_length": 10,
            "hash_ids": [1, 2],
        },
        # Valid: 64-token prompt -> ceil((64 + 10) / 64) = 2 blocks, fits.
        {
            "timestamp": 0.0,
            "input_length": 64,
            "output_length": 10,
            "hash_ids": [3],
        },
    ]
    trace_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    return trace_path


def test_offline_replay_rejects_oversized_request_without_hanging(tmp_path):
    trace_path = _write_oversized_then_valid_trace(tmp_path)

    report = run_trace_replay(
        trace_path,
        extra_engine_args=_trtllm_reject_args(),
        num_workers=1,
        replay_concurrency=1,  # the rejected head must free the slot or this hangs
        replay_mode="offline",
        trace_block_size=512,
    )

    assert report["num_requests"] == 2, "both requests arrived"
    assert report["completed_requests"] == 1, (
        "only the valid follower completes; the oversized request is terminally "
        "rejected and excluded from the report"
    )
    assert (
        report["total_output_tokens"] == 10
    ), "rejected request contributes no output tokens"
