# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from pathlib import Path

from benchmarks import mocker_request_enrich_trace


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _run_enrichment(
    monkeypatch,
    tmp_path: Path,
    direct_rows: list[dict],
    server_rows: list[dict],
) -> list[dict]:
    direct_path = tmp_path / "direct.jsonl"
    server_path = tmp_path / "server.jsonl"
    output_path = tmp_path / "enriched.jsonl"
    _write_jsonl(direct_path, direct_rows)
    _write_jsonl(server_path, server_rows)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mocker_request_enrich_trace.py",
            str(direct_path),
            str(server_path),
            str(output_path),
            "--strict",
            "--no-candidate-prefixes",
        ],
    )

    mocker_request_enrich_trace.main()
    return [json.loads(line) for line in output_path.read_text().splitlines()]


def test_enrichment_prefers_shared_request_id_over_prompt_hash(
    monkeypatch, tmp_path: Path
) -> None:
    shared_id = "bc4e1b7f-f25a-4599-8224-b1f6f61fe8e1"
    direct_row = {
        "tokens": [1, 2, 3],
        "max_output_tokens": 2,
        "uuid": shared_id,
        "request_id": shared_id,
        "dp_rank": 0,
        "arrival_timestamp_ms": None,
        "prompt_token_hash": "rollout-prompt",
        "generation_token_hash": "same-generation",
        "output_length": 2,
    }
    server_row = {
        "request_id": shared_id,
        "arrival_timestamp_ms": 123.0,
        "completion_timestamp_ms": 456.0,
        "status_code": 200,
        "worker_id": "worker-3",
        "chosen_worker": "worker-3",
        "prompt_token_hash": "server-prompt",
        "generation_token_hash": "same-generation",
        "output_length": 2,
    }

    rows = _run_enrichment(monkeypatch, tmp_path, [direct_row], [server_row])

    assert rows[0]["server_trace_matched"] is True
    assert rows[0]["server_trace_match_method"] == "request_id"
    assert rows[0]["server_trace_attempt_count"] == 1
    assert rows[0]["arrival_timestamp_ms"] == 123.0
    assert rows[0]["chosen_worker"] == "worker-3"
    assert rows[0]["server_prompt_hash_mismatch"] is True


def test_enrichment_selects_successful_retry_for_shared_request_id(
    monkeypatch, tmp_path: Path
) -> None:
    shared_id = "8f8bd743-eb2b-403c-9567-8e6c5f37bcc0"
    direct_row = {
        "tokens": [1],
        "max_output_tokens": 1,
        "uuid": shared_id,
        "request_id": shared_id,
        "dp_rank": 0,
        "arrival_timestamp_ms": None,
    }
    server_rows = [
        {
            "request_id": shared_id,
            "arrival_timestamp_ms": 10.0,
            "completion_timestamp_ms": 20.0,
            "status_code": 500,
            "error": "temporary failure",
            "worker_id": "worker-1",
            "chosen_worker": "worker-1",
        },
        {
            "request_id": shared_id,
            "arrival_timestamp_ms": 30.0,
            "completion_timestamp_ms": 40.0,
            "status_code": 200,
            "error": None,
            "worker_id": "worker-2",
            "chosen_worker": "worker-2",
        },
    ]

    rows = _run_enrichment(monkeypatch, tmp_path, [direct_row], server_rows)

    assert rows[0]["server_trace_match_method"] == "request_id"
    assert rows[0]["server_trace_attempt_count"] == 2
    assert rows[0]["arrival_timestamp_ms"] == 30.0
    assert rows[0]["chosen_worker"] == "worker-2"


def test_enrichment_keeps_hash_fallback_for_legacy_traces(
    monkeypatch, tmp_path: Path
) -> None:
    direct_row = {
        "tokens": [1, 2],
        "max_output_tokens": 1,
        "uuid": "baf1dcea-17c4-4bb2-9a5d-e27791b0c3d9",
        "dp_rank": 0,
        "arrival_timestamp_ms": None,
        "prompt_token_hash": "same-prompt",
        "generation_token_hash": "same-generation",
    }
    server_row = {
        "request_id": "unrelated-server-id",
        "arrival_timestamp_ms": 50.0,
        "status_code": 200,
        "prompt_token_hash": "same-prompt",
        "generation_token_hash": "same-generation",
    }

    rows = _run_enrichment(monkeypatch, tmp_path, [direct_row], [server_row])

    assert rows[0]["server_trace_matched"] is True
    assert rows[0]["server_trace_match_method"] == "trace_join_key"
    assert rows[0]["arrival_timestamp_ms"] == 50.0
