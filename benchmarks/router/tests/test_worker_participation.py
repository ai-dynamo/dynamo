# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pytest

from benchmarks.router import common
from benchmarks.router.common import (
    collect_worker_participation,
    get_common_aiperf_flags,
    validate_worker_participation,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.parallel,
]


def _record(phase: str, responses: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "metadata": {"benchmark_phase": phase},
        "status": 200,
        "responses": responses,
    }


def _write_raw_export(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as raw_export:
        for record in records:
            raw_export.write(json.dumps(record) + "\n")


def test_participation_flags_request_worker_ids_and_raw_export() -> None:
    flags = get_common_aiperf_flags(
        capture_worker_participation=True,
        nvext={"agent_hints": {"osl": 64}},
    )

    export_index = flags.index("--export-level")
    assert flags[export_index + 1] == "raw"

    extra_inputs = [
        flags[index + 1] for index, flag in enumerate(flags) if flag == "--extra-inputs"
    ]
    assert len(extra_inputs) == 2
    assert json.loads(extra_inputs[-1]) == {
        "nvext": {
            "agent_hints": {"osl": 64},
            "extra_fields": ["worker_id"],
        }
    }


def test_agent_dataset_preserves_per_request_nvext(tmp_path, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "common", common)
    agent_benchmark = importlib.import_module("benchmarks.router.agent_benchmark")

    source = tmp_path / "source.jsonl"
    output = tmp_path / "prepared.jsonl"
    source.write_text(
        json.dumps(
            {
                "session_id": "session-1",
                "input_length": 16,
                "output_length": 8,
                "extra": {
                    "nvext": {
                        "agent_hints": {"priority": 7},
                        "extra_fields": ["routing_state"],
                    }
                },
            }
        )
        + "\n"
    )

    prepared_path = agent_benchmark.prepare_dataset(
        source,
        output,
        capture_worker_participation=True,
    )

    prepared = json.loads(Path(prepared_path).read_text())
    assert prepared["extra"]["nvext"] == {
        "agent_hints": {"priority": 7},
        "extra_fields": ["routing_state", "worker_id"],
    }


def test_collect_worker_participation_reads_streaming_and_text_responses(
    tmp_path: Path,
) -> None:
    raw_export = tmp_path / "nested" / "profile_export_raw.jsonl"
    _write_raw_export(
        raw_export,
        [
            _record(
                "warmup",
                [
                    {
                        "text": json.dumps(
                            {"nvext": {"worker_id": {"decode_worker_id": 99}}}
                        )
                    }
                ],
            ),
            _record(
                "profiling",
                [
                    {
                        "packets": [
                            {
                                "name": "data",
                                "value": json.dumps(
                                    {
                                        "nvext": {
                                            "worker_id": {
                                                "prefill_worker_id": 11,
                                                "decode_worker_id": 21,
                                            }
                                        }
                                    }
                                ),
                            },
                            {"name": "data", "value": "[DONE]"},
                        ]
                    }
                ],
            ),
            _record(
                "profiling",
                [
                    {
                        "text": json.dumps(
                            {"nvext": {"worker_id": {"decode_worker_id": 22}}}
                        )
                    }
                ],
            ),
        ],
    )

    report = collect_worker_participation(tmp_path)

    assert report["profiling_requests"] == 2
    assert report["requests_with_worker_id"] == 2
    assert report["observed_prefill_worker_ids"] == [11]
    assert report["observed_decode_worker_ids"] == [21, 22]


def test_validate_worker_participation_writes_report_before_failure(
    tmp_path: Path,
) -> None:
    raw_export = tmp_path / "profile_export_raw.jsonl"
    _write_raw_export(
        raw_export,
        [
            _record(
                "profiling",
                [
                    {
                        "text": json.dumps(
                            {"nvext": {"worker_id": {"decode_worker_id": 21}}}
                        )
                    }
                ],
            )
        ],
    )

    with pytest.raises(RuntimeError, match="observed 1 decode workers, required 2"):
        validate_worker_participation(
            tmp_path,
            minimum_prefill_workers=0,
            minimum_decode_workers=2,
            logger=logging.getLogger(__name__),
        )

    report = json.loads((tmp_path / "worker_participation.json").read_text())
    assert report["observed_decode_worker_ids"] == [21]
    assert report["minimum_decode_workers"] == 2
