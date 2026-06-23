# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from benchmarks.exgentic_trace.convert_to_mooncake import main

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


def _span(start, end, input_tokens, output_tokens, status=1):
    return {
        "start_time": start,
        "end_time": end,
        "type": "llm_call",
        "status": {"code": status},
        "attributes": {
            "gen_ai.usage.input_tokens": input_tokens,
            "gen_ai.usage.output_tokens": output_tokens,
        },
    }


def test_converts_exgentic_parquet_to_session_mooncake(tmp_path):
    source = tmp_path / "trace.parquet"
    output = tmp_path / "trace.jsonl"
    row = {
        "session_id": "session-1",
        "spans": [
            _span("2026-01-01T00:00:00Z", "2026-01-01T00:00:10Z", 600, 10),
            _span("2026-01-01T00:00:03Z", "2026-01-01T00:00:04Z", 0, 0, 2),
            _span("2026-01-01T00:00:05Z", "2026-01-01T00:00:08Z", 700, 20),
            _span("2026-01-01T00:00:20Z", "2026-01-01T00:00:21Z", 100, 5),
        ],
    }
    pq.write_table(pa.Table.from_pylist([row]), source)

    assert main([str(tmp_path), "--output", str(output), "--block-size", "512"]) == 0

    converted = [json.loads(line) for line in output.read_text().splitlines()]
    assert [item["input_length"] for item in converted] == [600, 700, 100]
    assert [item["output_length"] for item in converted] == [10, 20, 5]
    assert [item["hash_ids"] for item in converted] == [[1, 2], [1, 2], [3]]
    assert converted[0]["timestamp"] == 1_767_225_600_000
    assert converted[1]["delay"] == 0
    assert converted[2]["delay"] == 12_000
